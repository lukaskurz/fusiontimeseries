import numpy as np
from sklearn.utils.extmath import _incremental_mean_and_var  # type: ignore
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_array, check_X_y, gen_batches
import warnings


class IncrementalLDA:
    """Incremental Linear Discriminant Analysis (ILDA).

    Incremental Linear Discriminant Analysis is a discriminant model that can be updated as datasets arrive.
    Alternatively it can also be used to improve regular Linear Discriminant Analysis by splitting the inputs
    into batches using the parameter batch_size
    """

    def __init__(self, unique_classes, priors=None, n_components=None, batch_size=None):
        self.priors = priors
        self.n_components = n_components
        self.batch_size = batch_size
        self.classes_ = unique_classes

    def predict(self, X):
        X = check_array(X, dtype=[np.float64, np.float32])
        n_inputs, n_features = X.shape
        # Have to improve
        # check_is_fitted(self)
        if not hasattr(self, "within_scatter"):
            raise Exception("Model has not been trained yet")

        print(self.within_scatter)
        print(self.between_scatter)

        ldaVec = self._get_lda_vecs()
        updatedX = np.dot(X, ldaVec)
        updated_class_means = np.dot(self.class_mean_, ldaVec)

        yVals = np.zeros((n_inputs,))

        for i in np.arange(n_inputs):
            currX = np.reshape(updatedX[i, :], (1, n_features))
            distVal = np.linalg.norm(np.subtract(updated_class_means, currX), axis=1)

            yVals[i] = self.classes_[distVal == distVal.min()]

        print(
            "The given point(s) belong to the following class(es) in the same order: ",
            yVals,
        )
        return yVals

    def _get_lda_vecs(self):
        lda_matrix = np.dot(np.linalg.pinv(self.within_scatter), self.between_scatter)
        eigVals, eigVecs = np.linalg.eigh(lda_matrix)
        sorted = np.argsort(eigVals)[::-1]
        eigVals = eigVals[sorted]
        ldaVec = eigVecs[:, sorted][:, : self.n_components].T
        eigVals = eigVals[: self.n_components]
        ldaVec /= np.linalg.norm(ldaVec, axis=0)
        return ldaVec, eigVals

    def fit(self, X, y):
        # Test if single fit or multiple fit
        X, y = check_X_y(X, y, estimator=self, ensure_min_samples=1)  # type: ignore
        self.classes_ = np.sort(unique_labels(y))

        if self.priors is None:  # estimate priors from sample
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            self.priors_ = np.bincount(y_t) / float(len(y))
        else:
            self.priors_ = np.asarray(self.priors)

        if (self.priors_ < 0).any():
            raise ValueError("priors must be non-negative")
        if not np.isclose(self.priors_.sum(), 1.0):
            warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        # Get the maximum number of components
        if self.n_components is None:
            self._max_components = len(self.classes_) - 1
        else:
            self._max_components = min(len(self.classes_) - 1, self.n_components)

        # LDA Logic begins here
        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size = 5 * n_features

        for batch in gen_batches(n_samples, self.batch_size):
            self.partial_fit(X[batch], y[batch])

        return self

    def partial_fit(self, X, y, check_input=False):
        n_samples, n_features = X.shape

        # print('X.shape', X.shape)
        # This is the first partial_fit
        if not hasattr(self, "n_samples_seen_"):
            self.n_samples_seen_ = 0
            self.class_n_samples_seen_ = np.zeros(self.classes_.shape)

            self.mean_ = np.zeros((1, n_features))
            self.class_mean_ = np.zeros((np.size(self.classes_), n_features))

            self.var_ = 0.0

            self.between_scatter = np.zeros((n_features, n_features))
            self.within_scatter = np.zeros((n_features, n_features))
            self.class_within_scatter = np.zeros(
                (n_features, n_features, self.classes_.size)
            )

        # If the number of samples is more than 1, we use a batch fit algorithm as in the reference paper Pang et al.
        if n_samples > 1:
            self._batch_fit(X, y, check_input)
        # Else if there is only 1 sample, we use a single fit algorithm as in the reference paper Pang et al.
        else:
            raise NotImplementedError("More than one sample is required for fitting")
        return self

    def _batch_fit(self, X, y, check_input=False):
        # print('Batch fit')
        if check_input:
            X, y = check_X_y(X, y, ensure_min_samples=2, estimator=self)  # type: ignore

        current_n_samples, n_features = X.shape
        # Update stats - they are 0 if this is the first step
        updated_mean, updated_var, updated_n_samples_seen_ = _incremental_mean_and_var(
            X,
            last_mean=self.mean_,
            last_variance=self.var_,
            last_sample_count=self.n_samples_seen_,
        )

        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X = np.subtract(X, updated_mean)
        else:
            col_batch_mean = np.mean(X, axis=0)
            X = np.subtract(X, col_batch_mean)

        # Updating algorithm
        # First update class means
        updated_class_mean = self.class_mean_
        updated_class_n_samples_seen_ = self.class_n_samples_seen_
        # print('updated_class_n_samples_seen_', updated_class_n_samples_seen_)
        # print('updated_class_mean', updated_class_mean)
        for i, current_class in enumerate(self.classes_):
            current_class_samples = X[y == current_class, :]
            n_current_class_samples = current_class_samples.shape[0]
            previous_n_class_samples = updated_class_n_samples_seen_[i]
            if n_current_class_samples > 0 and previous_n_class_samples > 0:
                previous_class_sum_current_class = (
                    updated_class_mean[i, :] * updated_class_n_samples_seen_[i]
                )
                current_class_sum_current_class = np.sum(current_class_samples, axis=0)

                # print('previous_class_sum_current_class.shape', previous_class_sum_current_class.shape)
                # print('current_class_sum_current_class.shape', current_class_sum_current_class.shape)
                # print('updated_class_mean.shape', updated_class_mean.shape)
                # print('updated_class_n_samples_seen_.shape', updated_class_n_samples_seen_[i])

                updated_class_n_samples_seen_[i] += n_current_class_samples
                updated_class_mean[i, :] = (
                    previous_class_sum_current_class + current_class_sum_current_class
                ) / previous_n_class_samples
            elif n_current_class_samples > 0:
                updated_class_mean[i, :] = np.mean(current_class_samples, axis=0)
                updated_class_n_samples_seen_[i] = n_current_class_samples

        # Then update between class scatter
        updated_between_scatter = self.between_scatter
        for i, current_class_mean in enumerate(updated_class_mean):
            n = X[y == self.classes_[i], :].shape[0]
            current_class_mean = current_class_mean.reshape(1, n_features)
            updated_mean = updated_mean.reshape(1, n_features)
            if n > 0:
                updated_between_scatter += n * (
                    current_class_mean - updated_mean
                ).T.dot(current_class_mean - updated_mean)

        # if np.any(np.isnan(updated_between_scatter)):
        #     print('Reached nan:::: ', n)
        #     print('Updatec class mean:::', updated_class_mean)
        #     print('updated mean::::', updated_mean)

        updated_class_within_scatter = self.class_within_scatter
        for i, current_class_mean in enumerate(updated_class_mean):
            current_class_samples = X[y == self.classes_[i], :]
            n_current_class_samples = current_class_samples.shape[0]
            l_c = current_class_samples.shape[0]
            n_c = self.class_n_samples_seen_[i]
            mean_y_c = np.reshape(
                np.mean(current_class_samples, axis=0), (n_features, 1)
            )

            if n_current_class_samples > 0 and n_c > 0:
                # print('current_class_samples.shape', current_class_samples.shape)
                mean_x_c = np.reshape(self.class_mean_[i, :], (n_features, 1))

                D_c = (mean_y_c - mean_x_c).dot((mean_y_c - mean_x_c).T)

                E_c = np.zeros(D_c.shape)
                for current_samples, j in enumerate(current_class_samples):
                    E_c += (current_samples - mean_x_c).dot(
                        (current_samples - mean_x_c).T
                    )

                F_c = np.zeros(D_c.shape)
                for current_samples, j in enumerate(current_class_samples):
                    F_c += (current_samples - mean_y_c).dot(
                        (current_samples - mean_y_c).T
                    )

                updated_class_within_scatter[:, :, i] += (
                    ((n_c * l_c * l_c) * D_c / np.square(n_c + l_c))
                    + ((np.square(n_c) * E_c) / np.square(n_c + l_c))
                    + ((l_c * (l_c + (2 * n_c)) * F_c) / np.square(n_c + l_c))
                )
            elif n_current_class_samples > 0:
                updated_class_within_scatter[:, :, i] = (
                    current_class_samples - mean_y_c
                ).dot((current_class_samples - mean_y_c).T)
        updated_within_scatter = np.sum(updated_class_within_scatter, axis=2)

        # Final values after computation
        self.n_samples_seen_ = updated_n_samples_seen_[0]
        self.class_n_samples_seen_ = updated_class_n_samples_seen_
        self.mean_ = updated_mean.squeeze()
        self.class_mean_ = updated_class_mean
        self.var_ = updated_var.squeeze()
        self.between_scatter = updated_between_scatter
        self.within_scatter = updated_within_scatter
        self.components_, self.eigenvalues_ = self._get_lda_vecs()
        self.class_within_scatter = updated_class_within_scatter
