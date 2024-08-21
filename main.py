import time, datetime
import tensorflow as tf
import network
import load_data
import warp
import util

class Project():
    def __init__(self):
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.labelN = 10
        self.warpN = 4
        self.epochs = 300

        # Load data & Initialize warp parameters
        self.train_datasets, self.val_datasets, self.test_datasets = load_data.load_mnist(batchSize=128)
        self.warp = warp.Process(imgSize='28x28', batchSize=128)

        # Create model
        self.Geometric_Predictor = network.Geometric_Predictor(warpDim=6)  # affine
        self.Classifier = network.Classifier(self.labelN)
        self.op_GP = tf.optimizers.SGD()
        self.op_Clf = tf.optimizers.SGD()

        # Create writer
        self.checkpoint_dir = f'./{self.current_time}/checkpoints'
        self.train_writer = tf.summary.create_file_writer(f'./{self.current_time}/logs/train')
        self.valid_writer = tf.summary.create_file_writer(f'./{self.current_time}/logs/valid')

        self.checkpoint = tf.train.Checkpoint(optimizer_GP=self.op_GP,
                                              optimizer_C=self.op_Clf,
                                              model_GP=self.Geometric_Predictor,
                                              model_C=self.Classifier)
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir, max_to_keep=30)


    def calculate_categorical_crossentropy(self, y_true, y_pred):
        labelOnehot = tf.one_hot(y_true, self.labelN)
        loss = tf.keras.losses.categorical_crossentropy(y_pred, labelOnehot)
        loss = tf.reduce_mean(loss)
        return loss

    def calculate_categorical_accuracy(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.uint8)
        eq_num = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        acc = tf.reduce_mean(eq_num).numpy()
        return acc

    def train_step(self, p, images, label):
        with tf.GradientTape() as GP_tape, tf.GradientTape() as C_tape:
            imageWarpAll = []
            pMtrxAll = []
            for l in range(self.warpN):
                pMtrx = self.warp.vec2mtrx(p)
                imageWarp= self.warp.transformImage(images, pMtrx)
                imageWarpAll.append(imageWarp)
                pMtrxAll.append(pMtrx)
                dp = self.Geometric_Predictor.call(imageWarp)
                p = self.warp.compose(p, dp)
            pMtrx = self.warp.vec2mtrx(p)
            imageWarp = self.warp.transformImage(images, pMtrx)
            imageWarpAll.append(imageWarp)
            pMtrxAll.append(pMtrx)
            imageWarp = imageWarpAll[-1]
            predictions = self.Classifier.call(imageWarp)
            loss = self.calculate_categorical_crossentropy(label, predictions)
            acc = self.calculate_categorical_accuracy(label, tf.argmax(predictions, axis=-1))
        gradients_of_GP = GP_tape.gradient(loss, self.Geometric_Predictor.trainable_variables)
        gradients_of_C = C_tape.gradient(loss, self.Classifier.trainable_variables)
        self.op_GP.apply_gradients(zip(gradients_of_GP, self.Geometric_Predictor.trainable_variables))
        self.op_Clf.apply_gradients(zip(gradients_of_C, self.Classifier.trainable_variables))
        return loss, acc, imageWarpAll, pMtrxAll, predictions

    def test_step(self, p, images, label):
        imageWarpAll = []
        pMtrxAll = []

        for l in range(self.warpN):
            pMtrx = self.warp.vec2mtrx(p)
            imageWarp = self.warp.transformImage(images, pMtrx)
            imageWarpAll.append(imageWarp)
            pMtrxAll.append(pMtrx)
            dp = self.Geometric_Predictor.call(imageWarp)
            p = self.warp.compose(p, dp)

        pMtrx = self.warp.vec2mtrx(p)
        imageWarp = self.warp.transformImage(images, pMtrx)
        imageWarpAll.append(imageWarp)
        pMtrxAll.append(pMtrx)

        imageWarp = imageWarpAll[-1]
        predictions = self.Classifier.call(imageWarp)
        loss = self.calculate_categorical_crossentropy(label, predictions)
        acc = self.calculate_categorical_accuracy(label, tf.argmax(predictions, axis=-1))
        return loss, acc, imageWarpAll, pMtrxAll, predictions

    def run(self):
        util.makedirs(f'./{self.current_time}/vis_warp_process/train')
        util.makedirs(f'./{self.current_time}/vis_warp_process/valid')
        for epoch in range(self.epochs):
            start = time.time()
            # ---------training---------
            L_tr = []
            A_tr = []
            for image_batch, label_batch in self.train_datasets:

                init_p = self.warp.genPerturbations(image_batch.get_shape()[0])
                loss_tr, acc_tr, imageWarpAll, pMtrxAll, _ = self.train_step(init_p, image_batch, label_batch)

                if len(L_tr) == 0:
                    self.warp.visualize_warp_process(image_batch, imageWarpAll, pMtrxAll, self.warpN,
                                                     f'./{self.current_time}/vis_warp_process/train/epoch-{str(epoch).zfill(3)}')
                L_tr.append(loss_tr)
                A_tr.append(acc_tr)

            # ---------validation---------
            L_val = []
            A_val = []
            for image_batch, label_batch in self.val_datasets:
                init_p = self.warp.genPerturbations(image_batch.get_shape()[0])
                loss_val, acc_val,imageWarpAll, pMtrxAll, _ = self.test_step(init_p, image_batch, label_batch)
                if len(L_val) == 0:
                    self.warp.visualize_warp_process(image_batch, imageWarpAll, pMtrxAll, self.warpN,
                                                     f'./{self.current_time}/vis_warp_process/valid/epoch-{str(epoch).zfill(3)}')
                L_val.append(loss_val)
                A_val.append(acc_val)



            with self.train_writer.as_default():
                tf.summary.scalar('loss(epoch)',  tf.reduce_mean(L_tr), step=epoch + 1)
                tf.summary.scalar('acc(epoch)',  tf.reduce_mean(A_tr), step=epoch + 1)

            with self.valid_writer.as_default():
                tf.summary.scalar('loss(epoch)', tf.reduce_mean(L_val), step=epoch + 1)
                tf.summary.scalar('acc(epoch)', tf.reduce_mean(A_val), step=epoch + 1)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            template = 'Train Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
            print(template.format(tf.reduce_mean(L_tr).numpy(), tf.reduce_mean(A_tr).numpy(),
                                  tf.reduce_mean(L_val).numpy(), tf.reduce_mean(A_val).numpy()))


            if (epoch + 1) % 10 == 0:
                self.ckpt_manager.save(epoch + 1)

        total = 0
        correct = 0
        for image_batch, label_batch in self.test_datasets:
            init_p = self.warp.genPerturbations(image_batch.get_shape()[0])
            _, _, _, pMtrxAll, predictions = self.test_step(init_p, image_batch, label_batch)
            equal = tf.cast(tf.equal(label_batch, tf.cast(tf.argmax(predictions, axis=-1), tf.uint8)), tf.float32)
            correct += tf.reduce_sum(equal)
            total += equal.get_shape()[0]
        acc = correct / total
        print(f'Acc of Testing Data:{acc}')

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus), gpus)

    project = Project()
    project.run()