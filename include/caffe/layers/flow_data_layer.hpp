#ifndef CAFFE_FLOW_DATA_LAYER_HPP_
#define CAFFE_FLOW_DATA_LAYER_HPP_

#include "hdf5.h"

#include <string>
#include <vector>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

#include "caffe/data_transformer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net from HDF5 files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */

template <typename Dtype>
class FlowBatch {
 public:
  Blob<Dtype> data_, label_, videoId_;
};

template <typename Dtype>
 class FlowDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit FlowDataLayer(const LayerParameter& param);
  virtual ~FlowDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reset();
  virtual void ThreadWork();

  virtual inline const char* type() const { return "FlowData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline vector<int> getLabels() const {return labels_;}
  virtual inline vector<int> getFullLabels() const {return full_labels_;}
  virtual inline vector<int> getVideoIds() const {return videoId_;}
  virtual inline vector<int> getMirrors() const {return mirrors_;}
  virtual inline vector<unsigned int> getCurrentPermutation() const {return file_permutation_;}

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual bool FillBatch(FlowBatch<Dtype>* batch);
  virtual void getBalancedPartition(const vector<string>& files, const vector<unsigned int>& subclasses, 
    const vector<int>& labels, vector<vector<bool> >& usedSamples, vector<string>& partition, vector<int>& labels_partition);
  virtual void readListFile(const string& source);
  virtual void TransformData(Dtype* data, const float dfactor, const float meanval, const int elements);
  virtual void showBlob(const Blob<Dtype>* top);

  vector<string> hdf_filenames_;
  bool output_labels_;
  unsigned int num_files_;
  unsigned int current_file_;
  vector<unsigned int> file_permutation_;

  FlowBatch<Dtype> prefetch_[5];
  BlockingQueue<FlowBatch<Dtype>*> prefetch_free_;
  BlockingQueue<FlowBatch<Dtype>*> prefetch_full_;

  boost::mutex mtx_;

  vector<vector<bool> > selected_files_;
  vector<unsigned int> subclasses_;
  vector<int> full_labels_;
  vector<int> labels_;
  vector<int> videoId_;
  vector<int> mirrors_;
  vector<string> full_hdf_filenames_;
  unsigned int balancingEpoch_;
  bool balance_;
  unsigned int prefetch_count_;
  boost::thread_group threads_;
  bool finish_threads_;
  bool do_balance_;
  unsigned int forw_files_;
  float meanval_;
  float dfactor_;
  unsigned int epoch_;
  unsigned int iter_;
  unsigned int init_balance_iteration_;
  int phase_;
};

}  // namespace caffe

#endif  // CAFFE_FLOW_DATA_LAYER_HPP_
