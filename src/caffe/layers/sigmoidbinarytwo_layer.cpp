#include <cmath>
#include <vector>

#include "caffe/layers/sigmoidbinarytwo_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoidbinarytwo(Dtype x,Dtype a) {
  return 2. / (1. + exp(-2*a*x)) - 1;
}

template <typename Dtype>
void SigmoidbinarytwoLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype t = this->layer_param_.sigmoidbinarytwo_param().t();
  Dtype e = this->layer_param_.sigmoidbinarytwo_param().e();
  Dtype a = log((2-t)/t)/e;
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoidbinarytwo(bottom_data[i],a);
  }
}

template <typename Dtype>
void SigmoidbinarytwoLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype t = this->layer_param_.sigmoidbinarytwo_param().t();
    Dtype e = this->layer_param_.sigmoidbinarytwo_param().e();
    Dtype a = log((2-t)/t)/e;
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = (top_data[i]+1) * 0.5;
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x)*2*a;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidbinarytwoLayer);
#endif

INSTANTIATE_CLASS(SigmoidbinarytwoLayer);
REGISTER_LAYER_CLASS(Sigmoidbinarytwo);


}  // namespace caffe
