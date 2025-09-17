from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import auc, classification_report, mean_absolute_error, mean_squared_error, precision_recall_curve, precision_recall_fscore_support, roc_curve
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import math
import warnings
import numpy as np
import plotly.graph_objects as go

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        fig = go.Figure()
        # Plot test_energy
        fig.add_trace(go.Scatter(
            y=test_energy,
            mode="lines",
            name="Test Energy"
        ))

        # Add threshold line
        fig.add_trace(go.Scatter(
            y=[threshold] * len(test_energy),
            mode="lines",
            line=dict(color="red", dash="dash"),
            name=f"Threshold = {threshold:.4f}"
        ))

        fig.update_layout(
            title="Test Energy vs Threshold",
            xaxis_title="Index",
            yaxis_title="Energy",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fig.write_html(f"TimesNet_test_energy_&_threshold_2000.html")
        fig.show()

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        
        print(classification_report(gt, pred, digits=4))

        f = open("result_anomaly_detection_TimesNet.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        
        self.plot_detection_results(test_data.data_x, pred, gt)
    
        mse = mean_squared_error(gt, pred)
        mae = mean_absolute_error(gt, pred)
        rmse = math.sqrt(mse)
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(gt, test_energy)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.4f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(
            title='Receiver Operating Characteristic (ROC)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        fig_roc.show()
        fig_roc.write_html(f"grafovi/TimesNet_ROC_curve.html")

        # Compute Precision-Recall curve and area
        precision_curve, recall_curve, _ = precision_recall_curve(gt, test_energy)
        pr_auc = auc(recall_curve, precision_curve)

        # Plot Precision-Recall curve
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode='lines', name=f'PR curve (AUC = {pr_auc:.4f})'))
        fig_pr.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            showlegend=True
        )
        fig_pr.show()
        fig_pr.write_html(f"grafovi/TimesNet_PR_curve.html")

        return
    
    def plot_detection_results(data, pred, gt):
        fig = go.Figure()

        # Signal
        fig.add_trace(go.Scatter(
            y=data,
            mode='lines',
            name=f'Signal'
        ))

        # Ground truth anomalies (green)
        fig.add_trace(go.Scatter(
            y=np.where(gt == 1, data.max(), np.nan),
            mode='markers',
            name='True Anomaly',
            marker=dict(color='green', size=7),
        ))

        # Detected false anomalies (orange)
        fig.add_trace(go.Scatter(
            y=np.where((pred == 1) & (gt == 0), data.max(), np.nan),
            mode='markers',
            name='False Positive',
            marker=dict(color='orange', size=7),
        ))

        # Detected anomalies (red)
        fig.add_trace(go.Scatter(
            y=np.where((pred == 1) & (gt == 1), data.max(), np.nan),
            mode='markers',
            name='Detected Anomaly',
            marker=dict(color='red', size=7),
        ))

        # Layout with axes starting at 0
        fig.update_layout(
            title='Anomaly Detection Results',
            xaxis_title='Time Step',
            yaxis_title='Signal Value',
            legend=dict(x=0, y=1.1, orientation="h"),
            xaxis=dict(range=[0, None]),
        )
        fig.show()
        fig.write_html(f"grafovi/TimesNet_anomaly_detection_result_2000.html")