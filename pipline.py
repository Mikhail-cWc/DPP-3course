import plotly.graph_objects as go
from torch.utils.data import DataLoader, Dataset
import wfdb
import torch
from scipy.signal import find_peaks
from sklearn.preprocessing import scale
import numpy as np
import json
import os
import plotly.io as io

class EcgPipelineDataset1D(Dataset):
    def __init__(self, path, mode=128):
        super().__init__()
        record = wfdb.rdrecord(path)
        self.signal = None
        self.mode = mode
        for sig_name, signal in zip(record.sig_name, record.p_signal.T):
            if sig_name in ["MLII", "II"] and np.all(np.isfinite(signal)):
                self.signal = scale(signal).astype("float32")
        if self.signal is None:
            raise Exception("No MLII LEAD")

        self.peaks = find_peaks(self.signal, distance=180)[0]
        mask_left = (self.peaks - self.mode // 2) > 0
        mask_right = (self.peaks + self.mode // 2) < len(self.signal)
        mask = mask_left & mask_right
        self.peaks = self.peaks[mask]

    def __getitem__(self, index):

        images = []
        peaks = []
        pred_peaks = 0
        for i in range(index, index + self.mode):
          peak_i = self.peaks[i]
          left_i, right_i = peak_i - self.mode // 2, peak_i + self.mode // 2
          img_i = self.signal[left_i:right_i]
          img_i = img_i.reshape(1, -1)
          images.append(img_i)
          peaks.append(img_i.argmax() + pred_peaks)
          pred_peaks += self.mode

        return {"image": torch.tensor(images), "peak": torch.tensor(peaks)}

    def get_dataloader(self, num_workers=4, batch_size=16, shuffle=True):
        data_loader = DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )
        return data_loader

    def __len__(self):
        return len(self.peaks)

class BasePipeline:
    def __init__(self, model, data_loader, path, beats = 30):
        self.path = path
        self.model = model
        self.beats = beats
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pipeline_loader = data_loader

        self.mapper = json.load(open("full_class_mapper.json"))

    def run_pipeline(self):
        self.model.eval()
        pd_class = np.empty(0)
        pd_peaks = np.empty(0)

        with torch.no_grad():
                batch = next(iter(self.pipeline_loader))
                inputs = batch["image"]
                inputs = torch.tensor(inputs)
                inputs = inputs.to(self.DEVICE)
                predictions = self.model(inputs)

                classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()
                pd_class = np.concatenate((pd_class, classes))
                pd_peaks = np.concatenate((pd_peaks, batch["peak"]))

        pd_class = pd_class.astype(int)[:self.beats]
        pd_peaks = pd_peaks.astype(int)[:self.beats]


        inputs = inputs.cpu().numpy()[:self.beats]
        inputs = inputs.squeeze(1).reshape(128*self.beats)
        annotations = []
        for i, (label, peak) in enumerate(zip(pd_class, pd_peaks)):
            if label != 0:
                annotations.append(
                    {
                        "x": peak,
                        "y": inputs[peak]-0.1,
                        "text": self.mapper.get(str(label)),
                        "xref": "x",
                        "yref": "y",
                        "showarrow": True,
                        "arrowcolor": "black",
                        "arrowhead": 1,
                        "arrowsize": 2,
                    },
                )

        fig = go.Figure(
            data=go.Scatter(
                x=list(range(len(inputs))),
                y=inputs,
            ),
        )
        fig.update_layout(
            title="ECG",
            xaxis_title="Time",
            yaxis_title="ECG Output Value",
            title_x=0.5,
            annotations=annotations,
            height=400,  # Set the desired height of the graph
            width=800, 
        )

        fig.write_json(
            os.path.join("./html/", os.path.basename(self.path + ".json")),
        )
        fig.write_image(os.path.join(
                 "./images/", os.path.basename(self.path + ".jpeg")
		))
        class_mapper = json.load(open("full_class_mapper.json"))
        pd_class = [class_mapper.get(str(value)) for value in pd_class]
        return pd_peaks, pd_class
