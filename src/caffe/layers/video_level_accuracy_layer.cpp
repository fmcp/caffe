#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <set>

#include "caffe/layers/video_level_accuracy_layer.hpp"

namespace caffe {

template <typename Dtype> VideoLevelAccuracyLayer<Dtype>::~VideoLevelAccuracyLayer() {
  computeAcc(labels_, rlabels_, videoIds_);
}

template <typename Dtype>
void VideoLevelAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void VideoLevelAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void VideoLevelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  vector<int> temp_labels, temp_rlabels, temp_videoIds;

  //Copy data.
  const Dtype* bottom_data0 = bottom[0]->cpu_data();
  const Dtype* bottom_data1 = bottom[1]->cpu_data();
  const Dtype* bottom_data2 = bottom[2]->cpu_data();
  for (int i=0;i<bottom[1]->count();i++) {
    labels_.push_back((int)bottom_data0[i]);
    rlabels_.push_back((int)bottom_data1[i]);
    videoIds_.push_back((int)bottom_data2[i]);

    temp_labels.push_back((int)bottom_data0[i]);
    temp_rlabels.push_back((int)bottom_data1[i]);
    temp_videoIds.push_back((int)bottom_data2[i]);
  }

  float acc = computeAcc(temp_labels, temp_rlabels, temp_videoIds);
  top[0]->mutable_cpu_data()[0] = acc;
  temp_labels.clear();
  temp_rlabels.clear();
  temp_videoIds.clear();
}

template <typename Dtype>
float VideoLevelAccuracyLayer<Dtype>::computeAcc(const vector<int>& estimLabs, const vector<int>& realLabs, const vector<int>& videoIds) {
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
  //LOG(INFO) <<  "Video level acc: " << (float)acc / (float)nvids;
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
