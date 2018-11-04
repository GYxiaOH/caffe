#include <algorithm>
#include <vector>
#include <iostream>

using namespace std;
#include "caffe/layers/hardth_layer.hpp"
#define DEBUG_AP(str) cout << str << endl;

namespace caffe {

template <typename Dtype>
void HardthLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype threshold = this->layer_param_.hardth_param().threshold();
  DEBUG_AP(threshold);
  for (int i = 0; i < count; ++i) {
    top_data[i] = (bottom_data[i] > 0?bottom_data[i]:- bottom_data[i])>threshold?bottom_data[i]:0;
  }
}

template <typename Dtype>
void HardthLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype threshold = this->layer_param_.hardth_param().threshold();
    DEBUG_AP(threshold);
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0?bottom_data[i]:- bottom_data[i]) > threshold);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(HardthLayer);
#endif

INSTANTIATE_CLASS(HardthLayer);
REGISTER_LAYER_CLASS(Hardth);

}  // namespace caffe
