#include <vector>

#include "caffe/layers/hamming_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HammingLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  diff1_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void HammingLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  const Dtype* bottom_data1 = bottom[0]->cpu_data();
  const Dtype* bottom_data2 = bottom[1]->cpu_data();
  caffe_cpu_onezero(count, bottom_data1, diff_.mutable_cpu_data());
  caffe_cpu_onezero(count, bottom_data2, diff1_.mutable_cpu_data());
  caffe_sub(
      count,
      diff_.cpu_data(),
      diff1_.cpu_data(),
      diff_.mutable_cpu_data());
  Dtype abssum = caffe_cpu_abs_sum(count,diff_.cpu_data());
  caffe_abs(count, diff_.cpu_data(), diff_.mutable_cpu_data());
  Dtype loss = abssum/bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void HammingLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype alpha = top[0]->cpu_diff()[0] / bottom[i]->num(); //loss_weight/n
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(HammingLossLayer);
#endif

INSTANTIATE_CLASS(HammingLossLayer);
REGISTER_LAYER_CLASS(HammingLoss);

}  // namespace caffe
