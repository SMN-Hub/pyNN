class LearningRateDecay:
    def __init__(self, alpha, delta):
        self.alpha0 = alpha
        self.delta = delta

    def next(self, iteration, epoch):
        return self.alpha0


class LearningRateDecayLinear(LearningRateDecay):
    def __init__(self, alpha, decay):
        super().__init__(alpha, decay)

    def next(self, iteration, epoch):
        return self.alpha0 / (1 + self.delta * epoch)


class LearningRateDecayExponential(LearningRateDecay):
    def __init__(self, alpha, delta):
        super().__init__(alpha, delta)

    def next(self, iteration, epoch):
        return self.alpha0 * pow(self.delta, epoch)


class LearningRateDecayRoot(LearningRateDecay):
    def __init__(self, alpha, delta):
        super().__init__(alpha, delta)

    def next(self, iteration, epoch):
        return self.alpha0 * self. delta / pow(epoch, 0.5) if epoch > 0 else self.alpha0


class LearningRateDecayDiscrete(LearningRateDecay):
    def __init__(self, alpha, thresholds):
        super().__init__(alpha, 0)
        self.thresholds = list(thresholds)

    def next(self, iteration, epoch):
        if iteration > 0 and len(self.thresholds) > 0 and iteration > self.thresholds[0]:
            self.delta += 1
            self.thresholds.pop(0)
            print("Learning rate down stairs")
        return self.alpha0 / (2 * self.delta) if self.delta > 0 else self.alpha0
