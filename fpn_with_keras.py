'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
from keras.layers import Input, Conv2D, Flatten, Dense, Activation, BatchNormalization
from keras.layers import Lambda, UpSampling2D, MaxPooling2D, Add
from keras.models import Sequential
from keras.models import Model
from keras.utils import plot_model


def ResNet_block(in_planes, planes, c_i_input, stride=1):
    """Build ResNet block
    Args:
        in_planes: the number of channels of input data
        planes: the number of channels of output data
        c_i_input: the input of ResNet block ,as c1, c2, c3, c4
        stride: convolution stride

    Returns:
        the output feature matching
    """
    expansion = 4
    conv1 = Conv2D(filters=planes, kernel_size=1, use_bias=False)(c_i_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(filters=planes, kernel_size=3, strides=stride, padding='same', use_bias=False)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv2D(filters=expansion*planes, kernel_size=1, use_bias=False)(conv2)
    conv3 = BatchNormalization()(conv3)
    # The shortcut of Resnet block
    shortcut = Sequential()
    if stride != 1 or in_planes != expansion*planes:
        shortcut.add(Conv2D(filters=expansion*planes, kernel_size=1, strides=stride, use_bias=False))
        shortcut.add(BatchNormalization())

    out = Add()([conv3, shortcut(c_i_input)])
    out = Activation('relu')(out)
    return out


class FPN():
    def __init__(self):
        """Build FPN
        Args:
            x: the input of the whole FPN
            num_blocks: a list of four elements, each element is the number of ResNet block of a stage
        """
        self.in_planes = 64
        # -----------------------------------------
        # Initialize some layers
        # -----------------------------------------

        # The first some layers with the input of FPN input and output of c1
        self.conv1 = Conv2D(filters=64, kernel_size=2, strides=2, padding='valid', use_bias=False)
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.maxpool1 = MaxPooling2D(pool_size=2, strides=2)

        # The layers with the input of c5 and output of p5
        self.toplayer = Conv2D(filters=256, kernel_size=1, strides=1, padding='valid')  # Reduce channels

        # Upsample layers
        self.upsample1 = UpSampling2D(size=(2, 2))
        self.upsample2 = UpSampling2D(size=(2, 2))
        self.upsample3 = UpSampling2D(size=(2, 2))

        # Lateral layers
        self.latlayer1 = Conv2D(filters=256, kernel_size=1, strides=1, padding='valid')
        self.latlayer2 = Conv2D(filters=256, kernel_size=1, strides=1, padding='valid')
        self.latlayer3 = Conv2D(filters=256, kernel_size=1, strides=1, padding='valid')

        # Smooth layer
        self.smooth1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.smooth2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.smooth3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')

    def _cal_c_i(self, planes, num_blocks, stride, c_i_input):
        """Build a stage ,then calculate c_i
        Args:
            planes: the number of channels of output data
            num_blocks: a list of four elements, each element is the number of ResNet block of a stage
            stride: convolution stride
            c_i_input: the input of ResNet block ,as c1, c2, c3, c4
        Returns:
            the output feature matching
        """
        # strides is a list with two elements of stride value
        strides = [stride] + [1]*(num_blocks-1)

        # Calculate the first ResNet block
        out = ResNet_block(self.in_planes, planes, c_i_input, strides[0])
        self.in_planes = planes * 4

        # Calculate the second ResNet block
        out = ResNet_block(self.in_planes, planes, out, strides[1])
        return out

    def _cal_p_i(self, x, y):
        """Calculate p_i

        Args:
          x: (Variable) feature map has been upsampled.
          y: (Variable) lateral feature map.

        Returns:
          p_i
        """
        _, H, W, _ = x.get_shape()
        _, H1, W1, _ = y.get_shape()
        if H == H1 and W == W1:
            print('The dimension of feature outputed by latlayer matches ' + '\n' +
                  'the dimension of feature outputed by upsampling layers')
            return Add()([x, y])

    def __call__(self, c_i_input, num_blocks):
        c1 = self.maxpool1(self.act1(self.bn1(self.conv1(c_i_input))))
        c2 = self._cal_c_i(64, num_blocks[0], 1, c1)
        c3 = self._cal_c_i(128, num_blocks[1], 2, c2)
        c4 = self._cal_c_i(256, num_blocks[2], 2, c3)
        c5 = self._cal_c_i(512, num_blocks[3], 2, c4)

        p5 = self.toplayer(c5)
        p4 = self._cal_p_i(self.upsample1(p5), self.latlayer1(c4))
        p3 = self._cal_p_i(self.upsample2(p4), self.latlayer2(c3))
        p2 = self._cal_p_i(self.upsample3(p3), self.latlayer3(c2))

        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return c1, c2, c3, c4, c5, p2, p3, p4, p5

inputs = Input(shape=(800, 800, 3))
# Create a instance of class FPN
net = FPN()
out = net(inputs, [2, 2, 2, 2])
model = Model(inputs=inputs, outputs=out[5:])

plot_model(model, show_shapes=True)
