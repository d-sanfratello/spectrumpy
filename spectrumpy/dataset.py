import numpy as np
import warnings


class DataSet:
    def __init__(self, lines, px, errpx, errlines, names):
        self.px = np.asarray(px)
        self.errpx = np.asarray(errpx)
        self.lines = np.asarray(lines)
        self.errlines = np.asarray(errlines)

        if isinstance(names, dict):
            self.names = names.copy()
        else:
            self.names = {}
            for i, l in enumerate(lines):
                self.names[l] = names[i]

        self.__sortdataset()

    def add_points(self, lines, px, errpx, errlines,  names):
        if not hasattr(lines, '__iter__'):
            raise TypeError("New points must be iterables.")
        if not hasattr(px, '__iter__'):
            raise TypeError("New points must be iterables.")
        if not hasattr(errpx, '__iter__'):
            raise TypeError("New points must be iterables.")
        if not hasattr(errlines, '__iter__'):
            raise TypeError("New points must be iterables.")

        if len(lines) != len(px) != len(errpx) != len(names) != len(errlines):
            raise ValueError(
                "Must insert the same amount of information for each"
                " datapoint."
            )

        lines = np.asarray(lines)
        errlines = np.asarray(errlines)
        px = np.asarray(px)
        errpx = np.asarray(errpx)

        self.px = np.concatenate((self.px, px))
        self.errpx = np.concatenate((self.errpx, errpx))
        self.lines = np.concatenate((self.lines, lines))
        self.errlines = np.concatenate((self.errlines, errlines))

        for i, l in enumerate(lines):
            self.names[l] = names[i]

        self.__sortdataset()

    def find_line(self, line_min, line_max):
        sup = self.lines[self.lines >= line_min]
        intr = sup[sup <= line_max]

        if len(intr) > 1:
            warnings.warn(
                "More than one wavelength found. Try reducing the interval "
                "if you need only one."
            )

            return intr
        elif len(intr) == 0:
            return None
        else:
            return intr[0]

    def find_line_idx(self, line=None, bounds=None):
        if line is None and bounds is not None:
            line_min, line_max = bounds

        if bounds is None:
            idx = np.where(self.lines == line)
            if np.any(idx):
                return idx
            else:
                return None
        else:
            line = self.find_line(line_min, line_max)
            if line is None or np.size(line) > 1:
                warnings.warn(f"{line}")
                return None
            else:
                return self.find_line_idx(line)

    def modify_point(self,
                     line, px, errpx=None, errline=None,
                     name=None):
        idx = np.where(self.lines == line)
        self.px[idx] = px

        if errpx is not None:
            self.errpx[idx] = errpx
        if errline is not None:
            self.errlines[idx] = errline
        if name is not None:
            self.names[line] = name

    def remove_point(self, line, inplace=True):
        idx = np.where(self.lines == line)

        if inplace:
            self.lines = np.delete(self.lines, idx)
            self.px = np.delete(self.px, idx)
            self.errpx = np.delete(self.errpx, idx)
            self.errlines = np.delete(self.errlines, idx)
            self.names.pop(line)
        else:
            aux_dict = self.names.copy()
            aux_dict.pop(line)
            return np.delete(self.lines, idx),\
                np.delete(self.px, idx),\
                np.delete(self.errpx, idx),\
                np.delete(self.errlines, idx)

    def __sortdataset(self):
        idx = np.argsort(self.lines)
        self.lines = np.take(self.lines, idx)
        self.px = np.take(self.px, idx)
        self.errpx = np.take(self.errpx, idx)
        self.errlines = np.take(self.errlines, idx)

    def __repr__(self):
        string = 'name\tLines [approx]\tsigma\tpx\terr\n'
        string += '--------------------------------------\n'
        for lin, s_l, px, s_px in zip(self.lines,
                                      self.errlines,
                                      self.px,
                                      self.errpx):
            string += f"{self.names[lin]}:" \
                      f"\t{lin} +- {s_l}\t\t{px:d} +- {s_px}\n"
        return string

    def __str__(self):
        return self.__repr__()
