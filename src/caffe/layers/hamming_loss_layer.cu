#include <vector>

#include "caffe/layers/hamming_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void HammingLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  const Dtype* bottom_data1 = bottom[0]->gpu_data();
  const Dtype* bottom_data2 = bottom[1]->gpu_data();
  caffe_gpu_onezero(count, bottom_data1, diff_.mutable_gpu_data());
  caffe_gpu_onezero(count, bottom_data2, diff1_.mutable_gpu_data());
  caffe_gpu_sub(
      count,
      diff_.gpu_data(),
      diff1_.gpu_data(),
      diff_.mutable_gpu_data());
  Dtype abssum ;
  caffe_gpu_asum(count,diff_.gpu_data(),&abssum);
  caffe_gpu_abs(count, diff_.gpu_data(), diff_.mutable_gpu_data());
  Dtype loss = abssum/bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void HammingLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype alpha = top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HammingLossLayer);

}  // namespace caffe
