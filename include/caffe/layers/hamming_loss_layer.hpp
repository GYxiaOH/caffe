#ifndef CAFFE_HAMMING_LOSS_LAYER_HPP_
#define CAFFE_HAMMING_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


template <typename Dtype>
class HammingLossLayer : public LossLayer<Dtype> {
 public:
  explicit HammingLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),diff_(),diff1_(){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "HammingLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.
  //virtual Dtype get_normalizer(
  //    LossParameter_NormalizationMode normalization_mode, int valid_count);
	Blob<Dtype> diff_;
        Blob<Dtype> diff1_;
 // int outer_num_;

 // Blob<Dtype> dots_;

  //LossParameter_NormalizationMode normalization_;
};

}  // namespace caffe

#endif // CAFFE_HAMMING_LOSS_LAYER_HPP_
