#include <cmath>
#include <vector>

#include "caffe/layers/sigmoidnew_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoidnew(Dtype x,Dtype a,Dtype e) {
  return 1. / (1. + exp(-a*(x-e/2)));
}

template <typename Dtype>
void SigmoidnewLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype t = this->layer_param_.sigmodnew_param().t();
  Dtype e = this->layer_param_.sigmodnew_param().e();
  Dtype a = 2*log((1-t)/t)/e;
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoidnew(bottom_data[i],a,e);
  }
}

template <typename Dtype>
void SigmoidnewLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype t = this->layer_param_.sigmodnew_param().t();
    Dtype e = this->layer_param_.sigmodnew_param().e();
    Dtype a = 2*log((1-t)/t)/e;
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x)*a;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidnewLayer);
#endif

INSTANTIATE_CLASS(SigmoidnewLayer);
REGISTER_LAYER_CLASS(Sigmoidnew);


}  // namespace caffe
