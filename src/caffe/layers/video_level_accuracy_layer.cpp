#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <set>

#include "caffe/layers/video_level_accuracy_layer.hpp"

namespace caffe {

template <typename Dtype> VideoLevelAccuracyLayer<Dtype>::~VideoLevelAccuracyLayer() {
  for (int i=0;i<labels_.size();i++) {
    LOG(INFO) << labels_[i] << " " << rlabels_[i] << " " << videoIds_[i];
  }
  computeAcc(labels_, rlabels_, videoIds_);
}

template <typename Dtype>
void VideoLevelAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ArgMaxParameter& argmax_param = this->layer_param_.argmax_param();
  out_max_val_ = argmax_param.out_max_val();
  top_k_ = argmax_param.top_k();
  has_axis_ = argmax_param.has_axis();
  CHECK_GE(top_k_, 1) << "top k must not be less than 1.";
  if (has_axis_) {
    axis_ = bottom[0]->CanonicalAxisIndex(argmax_param.axis());
    CHECK_GE(axis_, 0) << "axis must not be less than 0.";
    CHECK_LE(axis_, bottom[0]->num_axes()) <<
      "axis must be less than or equal to the number of axis.";
    CHECK_LE(top_k_, bottom[0]->shape(axis_))
      << "top_k must be less than or equal to the dimension of the axis.";
  } else {
    CHECK_LE(top_k_, bottom[0]->count(1))
      << "top_k must be less than or equal to"
        " the dimension of the flattened bottom blob per instance.";
  }
}

template <typename Dtype>
void VideoLevelAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int num_top_axes = bottom[0]->num_axes();
  if ( num_top_axes < 3 ) num_top_axes = 3;
  std::vector<int> shape(num_top_axes, 1);
  if (has_axis_) {
    // Produces max_ind or max_val per axis
    shape = bottom[0]->shape();
    shape[axis_] = top_k_;
  } else {
    shape[0] = bottom[0]->shape(0);
    // Produces max_ind
    shape[2] = top_k_;
    if (out_max_val_) {
      // Produces max_ind and max_val
      shape[1] = 2;
    }
  }
  top[0]->Reshape(shape);
}

template <typename Dtype>
void VideoLevelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  //Copy data.
  const Dtype* bottom_data0 = bottom[0]->cpu_data();
  const Dtype* bottom_data1 = bottom[1]->cpu_data();
  const Dtype* bottom_data2 = bottom[2]->cpu_data();
  for (int i=0;i<bottom[1]->count();i++) {
    labels_.push_back((int)bottom_data0[i]);
    rlabels_.push_back((int)bottom_data1[i]);
    videoIds_.push_back((int)bottom_data2[i]);
  }
}

template <typename Dtype>
void VideoLevelAccuracyLayer<Dtype>::computeAcc(const vector<int>& estimLabs, const vector<int>& realLabs, const vector<int>& videoIds) {
  // Get unique data.
  std::set<int> uvidsAux(videoIds.begin(), videoIds.end());
  vector<int> uvids(uvidsAux.begin(), uvidsAux.end());
  uvidsAux.clear();
  int nvids = uvids.size();

  // Find labels per video.
  vector<int> tlab(nvids);
  vector<int> rlab(nvids);
  vector<int> rlabs_;
  vector<int> tlabs_;
  for (int i = 0;i < nvids; i++) {
    for (int j = 0; j<videoIds.size(); j++) {
      if (videoIds[j] == uvids[i]) {
        rlabs_.push_back(realLabs[j]);
        tlabs_.push_back(estimLabs[j]);
      }
    } 

    rlab[i] = mode(rlabs_);
    tlab[i] = mode(tlabs_);

    rlabs_.clear();
    tlabs_.clear();
  }

  // Compute accuracy.
  int acc = 0;
  for (int i = 0; i < nvids; i++) {
    if (tlab[i] == rlab[i]) {
      acc++;
    }
  }
  LOG(INFO) <<  "Video level acc: " << (float)acc / (float)nvids;
}

template <typename Dtype>
int VideoLevelAccuracyLayer<Dtype>::mode(const vector<int>& data) {
  // Get unique data.
  std::set<int> dataAux(data.begin(), data.end());
  vector<int> kindData(dataAux.begin(), dataAux.end());
  vector<int> nsamples(kindData.size(), 0);

  // Count elements.
  int i = 0;
  for (vector<int>::iterator it=kindData.begin(); it!=kindData.end(); ++it) {
    nsamples[i] = std::count(data.begin(), data.end(), *it);
    i++;
  }

  // Get mode.
  int max = 0;
  for (int i = 1; i < nsamples.size(); i++) {
    if (nsamples[i] > nsamples[max]) {
      max = i;
    }
  }

  return kindData[max];
}

INSTANTIATE_CLASS(VideoLevelAccuracyLayer);
REGISTER_LAYER_CLASS(VideoLevelAccuracy);

}  // namespace caffe
