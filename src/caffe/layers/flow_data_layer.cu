/*
TODO:
- only load parts of the file, in accordance with a prototxt param "max_mem"
*/

#include <stdint.h>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/layers/flow_data_layer.hpp"

namespace caffe {

template <typename Dtype>
  void FlowDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    FlowBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
    // Transform data.
    Dtype* data = batch->data_.mutable_cpu_data();
    int elements = batch->data_.shape(0) * batch->data_.shape(1) * batch->data_.shape(2) * batch->data_.shape(3);
    TransformData(data, dfactor_, meanval_, elements);
    // Reshape to loaded data.
    top[0]->ReshapeLike(batch->data_);
    // Copy the data
    caffe_copy(batch->data_.count(), batch->data_.gpu_data(), top[0]->mutable_gpu_data());
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(), top[1]->mutable_gpu_data());
    if (phase_ == 1) { // Test time.
      // Reshape to loaded videoIds.
      top[2]->ReshapeLike(batch->videoId_);
      // Copy the videoIds.
      caffe_copy(batch->videoId_.count(), batch->videoId_.gpu_data(), top[2]->mutable_gpu_data());
    }
    // Ensure the copy is synchronous wrt the host, so that the next batch isn't
    // copied in meanwhile.
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    prefetch_free_.push(batch);

    iter_++;
    forw_files_ = forw_files_ + batch->data_.shape(0);
    if (forw_files_ == num_files_) {
      epoch_++;
      forw_files_ = 0;
      do_balance_ = true;
      Reset();
    }

    if (epoch_ % balancingEpoch_ == 0 && balance_ && do_balance_ && iter_ > init_balance_iteration_ && phase_ == 0) {
      LOG(INFO) << "New balanced partition";
      mtx_.lock();
      getBalancedPartition(full_hdf_filenames_, subclasses_, full_labels_, selected_files_, hdf_filenames_, labels_);
      num_files_ = hdf_filenames_.size();
      mtx_.unlock();
      do_balance_ = false;
    }
    showBlob(top[0]);
  }

  INSTANTIATE_LAYER_GPU_FORWARD(FlowDataLayer);

}  // namespace caffe
