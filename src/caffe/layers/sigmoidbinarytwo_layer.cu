#include <cmath>
#include <vector>

#include "caffe/layers/sigmoidbinarytwo_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SigmoidbinarytwoForward(const int n, const Dtype* in, Dtype* out,Dtype a) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 2. / (1. + exp(-2*a*in[index])) - 1;
  }
}

template <typename Dtype>
void SigmoidbinarytwoLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  Dtype t = this->layer_param_.sigmoidbinarytwo_param().t();
  Dtype e = this->layer_param_.sigmoidbinarytwo_param().e();
  Dtype a = log((2-t)/t)/e;
  SigmoidbinarytwoForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data,a);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void SigmoidbinarytwoBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff,Dtype a) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype sigmoid_x = (out_data[index]+1) * 0.5;
    out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x)*a;
  }
}

template <typename Dtype>
void SigmoidbinarytwoLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype t = this->layer_param_.sigmoidbinarytwo_param().t();
    Dtype e = this->layer_param_.sigmoidbinarytwo_param().e();
    Dtype a = log((2-t)/t)/e;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SigmoidbinarytwoBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_diff,a);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidbinarytwoLayer);


}  // namespace caffe
