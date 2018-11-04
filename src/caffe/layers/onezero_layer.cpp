#include <algorithm>
#include <vector>

#include "caffe/layers/onezero_layer.hpp"

namespace caffe {

template <typename Dtype>
void OnezeroLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    if(bottom_data[i]>Dtype(0.001))
    top_data[i] = 1;
    else if(bottom_data[i]<Dtype(0))
    top_data[i] = 0;
    else
    top_data[i] = 1000*bottom_data[i];
  }
}

template <typename Dtype>
void OnezeroLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
	if(bottom_data[i]>Dtype(0.001))
	    bottom_diff[i] = 0;
	else if(bottom_data[i]<Dtype(0))
	    bottom_diff[i] = 0;
	else
	    bottom_diff[i] = top_diff[i]*(1000);
  	}
  }
}


#ifdef CPU_ONLY
STUB_GPU(OnezeroLayer);
#endif

INSTANTIATE_CLASS(OnezeroLayer);
REGISTER_LAYER_CLASS(Onezero);

}  // namespace caffe
