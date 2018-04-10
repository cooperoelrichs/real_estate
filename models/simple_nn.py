from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras import backend as K

from sklearn.preprocessing import StandardScaler

from real_estate.models.price_model import PriceModel


class SimpleNeuralNetworkModel(object):
    LAMBDA_L2 = 0.05
    DROPOUT_FRACTION = 0.4

    def __init__(self, input_dim, nb_epoch, batch_size):
        self.input_dim = input_dim
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

    def compile_model(self):
        model = Sequential()
        model.add(Dense(
            input_dim=self.input_dim, output_dim=512,
            init='normal',
            activation='relu',
            W_regularizer=l2(self.LAMBDA_L2)
        ))
        model.add(Dropout(self.DROPOUT_FRACTION))
        model.add(Dense(
            input_dim=self.input_dim, output_dim=512,
            init='normal',
            activation='relu',
            W_regularizer=l2(self.LAMBDA_L2)
        ))
        model.add(Dropout(self.DROPOUT_FRACTION))
        model.add(Dense(
            input_dim=self.input_dim, output_dim=256,
            init='normal',
            activation='relu',
            W_regularizer=l2(self.LAMBDA_L2)
        ))
        model.add(Dropout(self.DROPOUT_FRACTION))
        model.add(Dense(
            input_dim=self.input_dim, output_dim=16,
            init='normal',
            activation='relu',
            W_regularizer=l2(self.LAMBDA_L2)
        ))
        model.add(Dropout(self.DROPOUT_FRACTION))
        model.add(Dense(
            output_dim=1,
            init='normal'
        ))

        adam = Adam(
            lr=0.002  # 0.001,
            # beta_1=0.9,
            # beta_2=0.999,
            # epsilon=None,
            # decay=0.0,
            # amsgrad=False
        )
        model.compile(
            loss='mean_squared_error',
            optimizer=adam,  # 'sgd',
            metrics=[SimpleNeuralNetworkModel.r2]
        )
        return model

    def fit(self, X_train, y_train):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_scaled = self.scaler.transform(X_train)

        self.model = self.compile_model()
        self.model.fit(
            X_scaled, y_train,
            nb_epoch=self.nb_epoch,
            batch_size=self.batch_size,
            verbose=2
            # validation_split=0.33
        )

    def predict(self, X_pred):
        X_scaled = self.scaler.transform(X_pred)
        return self.model.predict(
            X_scaled, batch_size=self.batch_size,
            verbose=0
        )[:, 0]

    def score(self, X_test, y_test):
        X_scaled = self.scaler.transform(X_test)
        loss_and_metrics = self.model.evaluate(
            X_scaled, y_test, batch_size=self.batch_size,
            verbose=0
        )
        return loss_and_metrics[1]

    def r2(y_true, y_pred):
        '''Keras r2 score.'''
        SS_res =  K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )


class NN(PriceModel):
    HAS_SIMPLE_COEFS = False
    HAS_FEATURE_IMPORTANCE = False
    MODEL_CLASS = SimpleNeuralNetworkModel

    PARAMS = {
        # 'input_dim': 10,
        'nb_epoch': 100,
        'batch_size': 1024,
    }

    def __init__(self, X, y, X_labels, params=None):
        if params is None:
            params = self.PARAMS
            params['input_dim'] = X.shape[1]
        super().__init__(X, y, X_labels, params)
