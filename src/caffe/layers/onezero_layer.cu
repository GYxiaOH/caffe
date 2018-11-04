#include <algorithm>
#include <vector>

#include "caffe/layers/onezero_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void OnezeroForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    if(in[index]>0.001)
        out[index] = 1;
    else
        out[index] = in[index] > 0 ? 1000*in[index] : 0;
  }
}

template <typename Dtype>
void OnezeroLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  OnezeroForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
__global__ void OnezeroBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    if(in_data[index]>0.001)
        out_diff[index] = 0;
    else
        out_diff[index] = in_data[index] > 0 ? 1000*in_diff[index] : 0;
  }
}

template <typename Dtype>
void OnezeroLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    OnezeroBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(OnezeroLayer);


}  // namespace caffe
