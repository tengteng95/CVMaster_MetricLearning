import argparse
from ast import parse
import os

import megengine
from loguru import logger
from megengine import functional as F
from megengine import module as M
from megengine import optimizer as optim
from megengine.autodiff import GradManager
from tqdm import tqdm

from datasets.build import make_dataloader
from loss.classification import CELoss
from model.resnet import resnet50
from utils.metrics import R1_mAP_eval


class ReIDBaseline(M.Module):
    def __init__(self, nr_class, feat_dim, droprate=0):
        super().__init__()
        self.backbone = resnet50(last_stride=1, pretrained=True)
        self.embedding = M.Linear(2048, feat_dim)
        self.bn_neck = M.BatchNorm1d(feat_dim)
        self.dropout = M.Dropout(droprate)
        self.criterion = CELoss(feat_dim, nr_class)

    def forward_one_step(self, x, labels):
        feat = self.backbone.extract_features(x)["res5"]
        feat = F.adaptive_avg_pool2d(feat, (1, 1))
        feat = F.flatten(feat, 1)

        embedding = self.embedding(feat)
        embedding = self.bn_neck(embedding)
        embedding = self.dropout(embedding)
        loss = self.criterion(embedding, labels)
        return loss

    def forward(self, x):
        feat = self.backbone.extract_features(x)["res5"]
        feat = F.adaptive_avg_pool2d(feat, oshp=(1, 1))
        feat = F.flatten(feat, 1)

        embedding = self.embedding(feat)
        embedding = self.bn_neck(embedding)

        return F.normalize(embedding, axis=1, ord=2)


def train(model, optimizer, gm, scheduler, train_loader, val_loader, num_query, args):
    best_metric = -1
    best_epoch = -1
    for epoch in range(args.max_epochs):
        model.train()
        scheduler.step()
        with tqdm(len(train_loader)) as t:
            t.set_description(f"[Epoch {epoch}/{args.max_epochs}]")
            for _, (imgs, labels, _, _, _) in enumerate(train_loader):
                imgs = megengine.tensor(imgs)
                labels = megengine.tensor(labels)
                optimizer.clear_grad()
                with gm:
                    loss = model.forward_one_step(imgs, labels)
                    gm.backward(loss)
                optimizer.step()
                t.set_postfix(
                    loss=f"{loss.item():6.2f}",
                    lr=f"{optimizer.param_groups[0]['lr']:6.5f}",
                )
                t.update()

        if (epoch + 1) % args.eval_period == 0:
            top1, mAP = eval(model, val_loader, num_query)
            current_metric = (top1 + mAP) * 0.5
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                best_save_path = os.path.join(args.output_dir, "best.pkl")
                megengine.save(
                    {"epoch": epoch, "state_dict": model.state_dict()}, best_save_path,
                )

            save_path = os.path.join(args.output_dir, "epoch_{}.pkl".format(epoch + 1))
            megengine.save(
                {"epoch": epoch, "state_dict": model.state_dict()}, save_path,
            )
            logger.info(f"Weights dumped to {save_path}...")
            logger.info(f"Best Metric is {best_metric} achieved at Epoch-{epoch}")


def eval(model, val_loader, num_query):
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=True)
    evaluator.reset()

    model.eval()
    img_path_list = []

    for img, pid, camid, _, imgpath in tqdm(val_loader, desc="Eval: Extract Features"):
        img = megengine.tensor(img)
        feat = model(img).to("cpux")
        evaluator.update((feat, pid, camid))
        img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], mAP


parser = argparse.ArgumentParser()


parser.add_argument("--data_root", type=str, default="./data")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--test-batch-size", type=int, default=256)
parser.add_argument("--num-instances", type=int, default=4)
parser.add_argument("--re-prob", type=float, default=0.5)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--feat-dim", type=int, default=2048)
parser.add_argument("--droprate", type=float, default=0)
parser.add_argument("--max-epochs", type=int, default=60)
parser.add_argument("--lr", default=0.02, type=float)
parser.add_argument("--weight-decay", default=0.0005, type=float)
parser.add_argument("--milestones", default=[40], nargs="+")
parser.add_argument("--eval-period", default=30, type=int)
parser.add_argument("--output-dir", default="./outputs")
parser.add_argument("--tag", default="CE_Baseline")

args = parser.parse_args()
args.milestones = [int(item) for item in args.milestones]
args.output_dir = os.path.join(args.output_dir, args.tag)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

train_loader, val_loader, num_query, num_classes, _, _ = make_dataloader(args)
model = ReIDBaseline(nr_class=num_classes, feat_dim=args.feat_dim, droprate=args.droprate)

logger.info(model)

optimizer = optim.SGD(model.parameters(), lr=args.lr)
scheduler = optim.MultiStepLR(optimizer, args.milestones)
gm = GradManager().attach(model.parameters())

train(model, optimizer, gm, scheduler, train_loader, val_loader, num_query, args)
