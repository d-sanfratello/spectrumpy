import numpy as np
import warnings


class DataSet:
    def __init__(self, lines, px, errpx, names):
        if not isinstance(px, np.ndarray):
            self.px = np.array(px)
        else:
            self.px = px.copy()

        if not isinstance(errpx, np.ndarray):
            self.errpx = np.array(errpx)
        else:
            self.errpx = errpx.copy()

        if not isinstance(lines, np.ndarray):
            self.lines = np.array(lines)
        else:
            self.lines = lines.copy()

        if isinstance(names, dict):
            self.names = names.copy()
        else:
            self.names = {}
            for i, l in enumerate(lines):
                self.names[l] = names[i]

        self.__sortdataset()

    def add_points(self, lines, px, errpx, names):
        try:
            _ = (item for item in lines)
            _ = (item for item in px)
            _ = (item for item in errpx)
            _ = (item for item in names)
        except TypeError:
            raise TypeError("New points must be iterables.")

        if len(lines) != len(px) != len(errpx) != len(names):
            raise ValueError(
                "Must insert the same amount of information for each "
                "datapoint."
            )

        self.px = np.concatenate((self.px, px))
        self.errpx = np.concatenate((self.errpx, errpx))
        self.lines = np.concatenate((self.lines, lines))

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
        elif len(intr) <= 0:
            return None
        else:
            return intr[0]

    def find_line_idx(self, line=None, line_min=None, line_max=None):
        if line is None and (line_min is None and line_max is None):
            raise ValueError(
                "Either `line` or bounds must be a wavelength."
            )

        if line is not None:
            idx = np.where(self.lines == line)
            if len(idx[0]) > 0:
                return idx
            else:
                return None
        else:
            line = self.find_line(line_min, line_max)
            if line is None:
                return line
            elif len(line) > 1:
                warnings.warn("{}".format(line))
                return None
            else:
                return self.find_line_idx(line)

    def modify_point(self, line, px, errpx=None, name=None):
        idx = np.where(self.lines == line)
        self.px[idx] = px

        if errpx is not None:
            self.errpx[idx] = errpx
        if name is not None:
            self.names[line] = name

    def remove_point(self, line, inplace=True):
        idx = np.where(self.lines == line)

        if inplace:
            self.lines = np.delete(self.lines, idx)
            self.px = np.delete(self.px, idx)
            self.errpx = np.delete(self.errpx, idx)
            self.names.pop(line)
        else:
            aux_dict = self.names.copy()
            aux_dict.pop(line)
            return np.delete(self.lines, idx),\
                np.delete(self.px, idx),\
                np.delete(self.errpx, idx),

    def __sortdataset(self):
        idx = np.argsort(self.lines)
        self.lines = np.take(self.lines, idx)
        self.px = np.take(self.px, idx)
        self.errpx = np.take(self.errpx, idx)

    def __repr__(self):
        string = 'Lines [approx]\tpx\terr\tname\n'
        string += '--------------------------------------\n'
        for i, l in enumerate(self.lines):
            string += "{:.2f}\t\t{:d}\t{:d}\t{:s}\n".format(
                l, self.px[i], self.errpx[i], self.names[l]
            )
        return string

    def __str__(self):
        return self.__repr__()
