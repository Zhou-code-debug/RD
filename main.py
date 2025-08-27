import torch, pickle, time, os
import numpy as np
import torch
from torch import nn
from param import args
from DataHander import DataHandler
from models.model import SDNet, GCNModel

from utils import load_model, save_model, fix_random_seed_as
from tqdm import tqdm

from models import diffusion_process as dp
from Utils.Utils import *
import logging
import sys
import warnings

class Coach:
    """训练器"""
    def __init__(self, handler):
        self.args = args
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        # 数据处理器
        self.handler = handler
        self.train_loader = self.handler.trainloader
        self.valloader = self.handler.valloader
        self.testloader = self.handler.testloader
        self.n_user, self.n_item = self.handler.n_user, self.handler.n_item
        # 图数据
        self.ui_Graph = self.handler.ui_graph.to(self.device)
        self.uu_Graph = self.handler.uu_graph.to(self.device)

        # 模型初始化
        self.GCNModel = GCNModel(args, self.n_user, self.n_item).to(self.device)

        # Build Diffusion process
        output_dims = [args.dims] + [args.n_hid]
        input_dims = output_dims[::-1]
        self.SDNet = SDNet(input_dims, output_dims, args.emb_size, time_type="cat", norm=args.norm).to(self.device)
        self.DiffProcess = dp.DiffusionProcess(args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max, args.steps, self.device).to(self.device)

        # 优化器
        self.optimizer1 = torch.optim.Adam([{'params': self.GCNModel.parameters(), 'weight_decay': 0}, ], lr=args.lr)
        self.optimizer2 = torch.optim.Adam([{'params': self.SDNet.parameters(), 'weight_decay': 0}, ], lr=args.difflr)

        # 学习率调度器
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(self.optimizer1, step_size=args.decay_step, gamma=args.decay)
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(self.optimizer2, step_size=args.decay_step, gamma=args.decay)

        # 训练历史
        self.train_loss = []
        self.his_recall = []
        self.his_ndcg = []

    def train(self):
        """完整训练流程"""
        print("开始训练RecDiff模型...")
        args = self.args
        self.save_history = True
        # 设置日志
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        log_save = './History/' + args.dataset + '/'
        log_file = args.save_name
        fname = f'{log_file}.txt'
        fh = logging.FileHandler(os.path.join(log_save, fname))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger()
        logger.addHandler(fh)
        logger.info(args)
        logger.info('================')
        best_recall, best_ndcg, best_epoch, wait = 0, 0, 0, 0
        start_time = time.time()

        for self.epoch in range(1, args.n_epoch + 1):
            # 训练一个epoch
            epoch_losses = self.train_one_epoch()
            self.train_loss.append(epoch_losses)
            print('epoch {} done! elapsed {:.2f}.s, epoch_losses {}'.format(self.epoch, time.time() - start_time, epoch_losses), flush=True)
            # 每5个epoch评估一次
            if self.epoch % 5 == 0:
                recall, ndcg = self.test(self.testloader)

                # Record the history of recall and ndcg
                self.his_recall.append(recall)
                self.his_ndcg.append(ndcg)
                # 检查是否是最佳结果
                cur_best = recall + ndcg > best_recall + best_ndcg
                if cur_best:
                    best_recall, best_ndcg, best_epoch = recall, ndcg, self.epoch
                    wait = 0
                else:
                    wait += 1
                logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}'.format(
                    self.epoch, time.time() - start_time, args.topk, recall, args.topk, ndcg))
                if args.model_dir and cur_best:
                    desc = args.save_name
                    perf = ''  # f'N/R_{ndcg:.4f}/{hr:.4f}'
                    fname = f'{args.desc}_{desc}_{perf}.pth'

                    save_model(self.GCNModel, self.SDNet, os.path.join(args.model_dir, fname), self.optimizer1, self.optimizer2)
            if self.save_history:
                self.saveHistory()

            # 早停检查
            if wait >= args.patience:
                print(f'Early stop at epoch {self.epoch}, best epoch {best_epoch}')
                break

        print(f'Best  Recall@{args.topk} {best_recall:.6f}, NDCG@{args.topk} {best_ndcg:.6f},', flush=True)

    def train_one_epoch(self):
        """训练一个epoch"""
        self.SDNet.train()
        self.GCNModel.train()
        dataloader = self.train_loader
        # epoch_losses = [0] * 3
        epoch_losses = {'bpr': 0.0, 'reg': 0.0, 'diff': 0.0}
        num_batches = 0

        # 负采样
        dataloader.dataset.negSampling()
        tqdm_dataloader = tqdm(dataloader)
        since = time.time()

        for iteration, batch in enumerate(tqdm_dataloader):
            user_idx, pos_idx, neg_idx = batch
            user_idx = user_idx.long().cuda()
            pos_idx = pos_idx.long().cuda()
            neg_idx = neg_idx.long().cuda()
            # 前向传播
            ui_embeds, uu_embeds = self.GCNModel(self.ui_Graph, self.uu_Graph, True)
            # 分离用户和物品嵌入
            u_embeds = ui_embeds[:self.n_user]
            i_embeds = ui_embeds[self.n_user:]
            # 获取batch嵌入
            user = u_embeds[user_idx]
            pos = i_embeds[pos_idx]
            neg = i_embeds[neg_idx]

            # 扩散损失计算
            uu_terms = self.DiffProcess.calculate_losses(self.SDNet, uu_embeds[user_idx], args.reweight)
            diff_loss = uu_terms["loss"].mean()

            # 添加去噪后的用户嵌入
            user = user + uu_terms["pred_xstart"]

            # BPR损失
            score_diff = pairPredict(user, pos, neg)
            bpr_loss = -torch.mean(torch.log(torch.sigmoid(score_diff)))

            # 正则化损失
            reg_loss = ((torch.norm(user) ** 2 + torch.norm(pos) ** 2 + torch.norm(neg) ** 2) * args.reg) / args.batch_size

            # 总损失
            total_loss = diff_loss + bpr_loss + reg_loss

            # losses = [bpr_loss.item(), reg_loss.item()]
            # losses.append(diff_loss.item())

            # 反向传播
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            total_loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            # 记录损失
            epoch_losses['bpr'] += bpr_loss.item()
            epoch_losses['reg'] += reg_loss.item()
            epoch_losses['diff'] += diff_loss.item()
            num_batches += 1

            # epoch_losses = [x + y for x, y in zip(epoch_losses, losses)]

        # 更新学习率
        self.scheduler1.step()
        self.scheduler2.step()

        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # epoch_losses = [sum(epoch_losses)] + epoch_losses
        time_elapsed = time.time() - since
        print('Training complete in {:.4f}s'.format(time_elapsed))
        return epoch_losses

    def _calculate_metrics(self, top_items, test_items, user_indices):
        """计算Recall和NDCG指标"""
        assert top_items.shape[0] == len(user_indices)
        total_recall = 0
        total_ndcg = 0

        for i in range(len(user_indices)):
            user_top_items = list(top_items[i])
            user_test_items = test_items[user_indices[i]]
            test_num = len(user_test_items)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(test_num, args.topk))])
            recall = dcg = 0
            for val in user_test_items:
                if val in user_top_items:
                    recall += 1
                    dcg += np.reciprocal(np.log2(user_top_items.index(val) + 2))
            recall = recall / test_num
            ndcg = dcg / maxDcg
            total_recall += recall
            total_ndcg += ndcg
        return total_recall, total_ndcg

    def test(self, dataloader):
        """评估模型性能"""
        self.SDNet.eval()
        self.GCNModel.eval()
        Recall, NDCG = [0] * 2
        num = dataloader.dataset.__len__()

        since = time.time()
        with torch.no_grad():
            # 获取所有嵌入
            ui_embeds, uu_embeds = self.GCNModel(self.ui_Graph, self.uu_Graph, True)
            tqdm_dataloader = tqdm(dataloader)
            for iteration, batch in enumerate(tqdm_dataloader, start=1):
                user_idx, trnMask = batch
                user_idx = user_idx.long().to(self.device)
                trnMask = trnMask.to(self.device)

                # 分离用户和物品嵌入
                u_embeds = ui_embeds[:self.n_user]
                i_embeds = ui_embeds[self.n_user:]
                # 获取当前批次的用户嵌入
                user = u_embeds[user_idx]
                uu_emb = uu_embeds[user_idx]

                # 社交去噪
                user_predict = self.DiffProcess.p_sample(self.SDNet, uu_emb, args.sampling_steps, args.sampling_noise)
                # 融合去噪后的社交信息
                user = user + user_predict
                # 计算所有物品的评分
                allPreds = t.mm(user, t.transpose(i_embeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
                # 获取Top-K推荐
                _, topLocs = t.topk(allPreds, args.topk)
                # 计算指标
                recall, ndcg = self._calculate_metrics(topLocs.cpu().numpy(), dataloader.dataset.tstLocs, user_idx)
                Recall += recall
                NDCG += ndcg
            time_elapsed = time.time() - since
            print('Testing complete in {:.4f}s'.format(time_elapsed))
            Recall = Recall / num
            NDCG = NDCG / num
        return Recall, NDCG

    def saveHistory(self):
        history = dict()
        history['loss'] = self.train_loss
        history['Recall'] = self.his_recall
        history['NDCG'] = self.his_ndcg
        ModelName = "SDR"
        desc = args.save_name
        perf = ''  # f'N/R_{ndcg:.4f}/{hr:.4f}'
        fname = f'{args.desc}_{desc}_{perf}.his'

        with open('./History/' + args.dataset + '/' + fname, 'wb') as fs:
            pickle.dump(history, fs)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    fix_random_seed_as(args.seed)

    handler = DataHandler()
    handler.LoadData()
    app = Coach(handler)
    app.train()
