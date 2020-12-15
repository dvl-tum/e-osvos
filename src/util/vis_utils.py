"""Visdom Visualization."""
import numpy as np
import torch
from visdom import Visdom
import logging


logging.getLogger('visdom').setLevel(logging.CRITICAL)


class BaseVis(object):

    def __init__(self, viz_opts, update_mode='append', env=None, win=None,
                 resume=False, port=8097, server='http://localhost'):
        self.viz_opts = viz_opts
        self.update_mode = update_mode
        self.win = win
        if env is None:
            env = 'main'

        import logging
        log_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        self.viz = Visdom(env=env, port=port, server=server)
        logging.getLogger().setLevel(log_level)

        # if resume first plot should not update with replace
        self.removed = not resume

    def win_exists(self):
        return self.viz.win_exists(self.win)

    def close(self):
        if self.win is not None:
            self.viz.close(win=self.win)
            self.win = None

    def register_event_handler(self, handler):
        self.viz.register_event_handler(handler, self.win)


class LineVis(BaseVis):
    """Visdom Line Visualization Helper Class."""

    def plot(self, y_data, x_label):
        """Plot given data.

        Appends new data to exisiting line visualization.
        """
        update = self.update_mode
        # update mode must be None the first time or after plot data was removed
        if self.removed:
            update = None
            self.removed = False

        if isinstance(x_label, list):
            Y = torch.Tensor(y_data)
            X = torch.Tensor(x_label)
        else:
            y_data = [d.cpu() if torch.is_tensor(d)
                      else torch.tensor(d)
                      for d in y_data]

            Y = torch.Tensor(y_data).unsqueeze(dim=0)
            X = torch.Tensor([x_label])

        win = self.viz.line(X=X, Y=Y, opts=self.viz_opts, win=self.win, update=update)

        if self.win is None:
            self.win = win
        self.viz.save([self.viz.env])

    def reset(self):
        #TODO: currently reset does not empty directly only on the next plot.
        # update='remove' is not working as expected.
        if self.win is not None:
            # self.viz.line(X=None, Y=None, win=self.win, update='remove')
            self.removed = True


class HeatVis(BaseVis):
    """Visdom heatmap visualization helper class."""

    def __init__(self, *args, **kwargs):
        super(HeatVis, self).__init__(*args, **kwargs)
        self.X_hist = []

    def plot(self, X):
        """Plot given data.

        Appends new data to exisiting surface visualization.
        """
        #if self.win is None:
        #    self.X_hist = []
        self.X_hist.append(X.clone().cpu())
        if len(self.X_hist) < 2:
            return

        X_hist = torch.stack(self.X_hist).numpy().astype(np.float64)

        self.win = self.viz.heatmap(
            X=X_hist,
            opts=self.viz_opts,
            win=self.win, )
        self.viz.save([self.viz.env])

    def close(self):
        super(HeatVis, self).close()
        self.X_hist = []

    def reset(self):
        self.X_hist = []


class ImgVis(BaseVis):
    """Visdom Image Visualization Helper Class."""

    def plot(self, images):
        """Plot given images."""

        images = [img.data if isinstance(img, torch.autograd.Variable)
                  else img for img in images]
        images = [img.squeeze(dim=0) if len(img.size()) == 4
                  else img for img in images]

        self.win = self.viz.images(
            tensor=images,
            opts=self.viz_opts,
            win=self.win, )
        self.viz.save([self.viz.env])


class TextVis(BaseVis):
    """Visdom Text Visualization Helper Class."""

    def plot(self, text):
        """Plot given text."""

        self.win = self.viz.text(text, opts=self.viz_opts, win=self.win, )
        self.viz.save([self.viz.env])

