/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
:: use util functions caffe_copy, and Blob->offset()
:: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <boost/thread/mutex.hpp>

#include <iostream>
#include <fstream>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layers/flow_data_layer.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"

#include "caffe/data_transformer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
  FlowDataLayer<Dtype>::FlowDataLayer(const LayerParameter& param)
  : BaseDataLayer<Dtype>(param), prefetch_free_(), prefetch_full_() {}

template <typename Dtype>
  FlowDataLayer<Dtype>::~FlowDataLayer<Dtype>() {
    finish_threads_ = true;
    threads_.join_all();
  }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
bool FlowDataLayer<Dtype>::FillBatch(FlowBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  bool stop = false;

  int batch_size = this->layer_param_.flow_data_param().batch_size();

  mtx_.lock();
  const int current_file = current_file_;
  current_file_ = current_file_ + batch_size;
  if (current_file_ > hdf_filenames_.size()) {
  	batch_size = hdf_filenames_.size() - current_file;
  	current_file_ = hdf_filenames_.size();

  	if (current_file == hdf_filenames_.size()) {
  		stop = true;
  	}
  }
  mtx_.unlock();

  // If we have read all files of the batch, we stop.
  if (!stop) {
  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  	hid_t file_id = H5Fopen(hdf_filenames_[0].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // Read a data point, and use it to initialize the top blob.
  	Blob<Dtype> sample_blob(1, 1, 1, 1);
  	hdf5_load_nd_dataset(file_id, this->layer_param_.top(0).c_str(), 1, INT_MAX, &sample_blob);
  	vector<int> top_shape(4);
  	vector<int> top_label_shape(2);
  	top_shape[0] = batch_size;
  	top_shape[1] = sample_blob.shape(0);
  	top_shape[2] = sample_blob.shape(1);
  	top_shape[3] = sample_blob.shape(2);
  	top_label_shape[0] = batch_size;
  	top_label_shape[1] = 1;
  	batch->data_.Reshape(top_shape);
  	batch->label_.Reshape(top_label_shape);
    batch->videoId_.Reshape(top_label_shape);
    const int blob_size = sample_blob.shape(0) * sample_blob.shape(1) * sample_blob.shape(2);

  	Dtype* top_data = batch->data_.mutable_cpu_data();
  	Dtype* top_label = batch->label_.mutable_cpu_data(); 
    if (phase_ == 1) { // Test time.
      Dtype* top_videoId = batch->videoId_.mutable_cpu_data(); 
    }

  	herr_t status = H5Fclose(file_id);
  	CHECK_GE(status, 0) << "Failed to close HDF5 file: " << hdf_filenames_[0];

  	for (int item_id = current_file, offset_ind = 0; item_id < current_file + batch_size; ++item_id, offset_ind++) {
  		timer.Start();
    // get files.
  		const char* filename
  		= hdf_filenames_[file_permutation_[item_id]].c_str();
  		DLOG(INFO) << "Loading HDF5 file: " << filename;
  		file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  		if (file_id < 0) {
  			LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  		}else {
    		hdf5_load_nd_dataset(file_id, this->layer_param_.top(0).c_str(), 1, INT_MAX, &sample_blob);
    		herr_t status = H5Fclose(file_id);
    		CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;
    		DLOG(INFO) << "Successully loaded " << filename;

        // Copy data.
    		caffe_copy(sample_blob.count(), sample_blob.cpu_data(), top_data + (blob_size * offset_ind));

        // Copy label.
    		top_label[item_id - current_file] = labels_[file_permutation_[item_id]];

        if (phase_ == 1) { // Test time.
          // Copy videoId.
          top_videoId[item_id - current_file] = videoId_[file_permutation_[item_id]];
        }

    		trans_time += timer.MicroSeconds();
      }
  	}

  	vector<int> top_info(batch->data_.shape());
  	DLOG(INFO) << "Batch data size: " << top_info[0] << ","
  	<< top_info[1] << "," << top_info[2] << "," << top_info[3];

  	timer.Stop();
  	batch_timer.Stop();
  	DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  	DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  	DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  }

  return stop;
}

template <typename Dtype>
  void FlowDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    // Initialize parameters.
    balance_ = this->layer_param_.flow_data_param().balance();
    balancingEpoch_ = this->layer_param_.flow_data_param().balancing_epoch();
    prefetch_count_ = this->layer_param_.flow_data_param().prefetch_count();
    meanval_ = this->layer_param_.flow_data_param().meanval();
    dfactor_ = 1.0 / this->layer_param_.flow_data_param().compress_factor();
    init_balance_iteration_ = this->layer_param_.flow_data_param().init_balance_iteration();
    phase_ = this->layer_param_.flow_data_param().phase();
    finish_threads_ = false;
    forw_files_ = 0;
    do_balance_ = false;
    epoch_ = 1;
    iter_ = 1;
    
    // Initialize queue.
    for (int i = 0; i < prefetch_count_; ++i) {
      prefetch_free_.push(&prefetch_[i]);
    }

    // Refuse transformation parameters since HDF5 is totally generic.
    CHECK(!this->layer_param_.has_transform_param()) <<
    this->type() << " does not transform data.";

    // Read the source to parse the filenames.
    const string& source = this->layer_param_.flow_data_param().source();
    readListFile(source);

    // Get balanced partition.
    if (balance_) {
      full_hdf_filenames_ = hdf_filenames_;
      full_labels_ = labels_;
      getBalancedPartition(full_hdf_filenames_, subclasses_, full_labels_, selected_files_, hdf_filenames_, labels_);
      num_files_ = hdf_filenames_.size();
    }

    file_permutation_.clear();
    file_permutation_.resize(num_files_);
    // Default to identity permutation.
    for (int i = 0; i < num_files_; i++) {
      file_permutation_[i] = i;
    }

    // Shuffle if needed.
    if (this->layer_param_.flow_data_param().shuffle()) {
      std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
    }

    // Reshape blobs.
    const int batch_size = this->layer_param_.flow_data_param().batch_size();
    hid_t file_id = H5Fopen(hdf_filenames_[0].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    // Read a data point, and use it to initialize the top blob.
    Blob<Dtype> sample_blob(1, 1, 1, 1);
    hdf5_load_nd_dataset(file_id, this->layer_param_.top(0).c_str(), 1, INT_MAX, &sample_blob);
    vector<int> top_shape(4);
    top_shape[0] = batch_size;
    top_shape[1] = sample_blob.shape(0);
    top_shape[2] = sample_blob.shape(1);
    top_shape[3] = sample_blob.shape(2);

    top[0]->Reshape(top_shape);

    for (int i = 0; i < this->prefetch_count_; ++i) {
      this->prefetch_[i].data_.Reshape(top_shape);
    }

    LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

    // label
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_count_; ++i) {
    	this->prefetch_[i].label_.Reshape(label_shape);
    }

    if (phase_ == 1) { // Test time.
      // videoId
      vector<int> videoId_shape(1, batch_size);
      top[2]->Reshape(videoId_shape);
      for (int i = 0; i < this->prefetch_count_; ++i) {
        this->prefetch_[i].videoId_.Reshape(videoId_shape);
      }
    }

    herr_t status = H5Fclose(file_id);
    CHECK_GE(status, 0) << "Failed to close HDF5 file: " << hdf_filenames_[0];
 
    Reset();

    DLOG(INFO) << "Initializing prefetch";

    for (int i=0; i<prefetch_count_;i++) {
    	threads_.add_thread(new boost::thread(boost::bind(&FlowDataLayer<Dtype>::ThreadWork, this)));
    }

    DLOG(INFO) << "Prefetch initialized.";
  }

template <typename Dtype>
  void FlowDataLayer<Dtype>::Reset() {
  	mtx_.lock();
    current_file_ = 0;
    mtx_.unlock();
  }

template <typename Dtype>
  void FlowDataLayer<Dtype>::ThreadWork() {
    #ifndef CPU_ONLY
      cudaStream_t stream;
      if (Caffe::mode() == Caffe::GPU) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      }
    #endif

    try {
      FlowBatch<Dtype>* batch;
      bool newBatch = false;
      while (!finish_threads_) {
        newBatch = prefetch_free_.try_pop(&batch);
        if (newBatch) {
          bool stop = FillBatch(batch);
          if (!stop) {
  	        #ifndef CPU_ONLY
              if (Caffe::mode() == Caffe::GPU) {
              batch->data_.data().get()->async_gpu_push(stream);
              CUDA_CHECK(cudaStreamSynchronize(stream));
              }
  	        #endif
            prefetch_full_.push(batch);
          }else {
            prefetch_free_.push(batch);
          }
        }
      }
    } catch (boost::thread_interrupted&) {
      // Interrupted exception is expected on shutdown
    }
    #ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        CUDA_CHECK(cudaStreamDestroy(stream));
      }
    #endif
  }

template <typename Dtype>
  void FlowDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    FlowBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
    // Transform data.
    Dtype* data = batch->data_.mutable_cpu_data();
    int elements = batch->data_.shape(0) * batch->data_.shape(1) * batch->data_.shape(2) * batch->data_.shape(3);
    TransformData(data, dfactor_, meanval_, elements);
    // Reshape to loaded data.
    top[0]->ReshapeLike(batch->data_);
    // Copy the data
    caffe_copy(batch->data_.count(), batch->data_.cpu_data(), top[0]->mutable_cpu_data());
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(), top[1]->mutable_cpu_data());
    if (phase_ == 1) { // Test time.
      // Reshape to loaded videoIds.
      top[2]->ReshapeLike(batch->videoId_);
      // Copy the videoIds.
      caffe_copy(batch->videoId_.count(), batch->videoId_.cpu_data(), top[2]->mutable_cpu_data());
    }

    prefetch_free_.push(batch);

    iter_++;

    forw_files_ = forw_files_ + batch->data_.shape(0);
    if (forw_files_ == num_files_) {
      epoch_++;
      forw_files_ = 0;
      do_balance_ = true;
      Reset();
    }

    if (epoch_ % balancingEpoch_ == 0 && balance_ && do_balance_ && iter_ > init_balance_iteration_) {
      LOG(INFO) << "New balanced partition";
      mtx_.lock();
      getBalancedPartition(full_hdf_filenames_, subclasses_, full_labels_, selected_files_, hdf_filenames_, labels_);
      num_files_ = hdf_filenames_.size();
      mtx_.unlock();
      do_balance_ = false;
    }
  }

template <typename Dtype>
  void FlowDataLayer<Dtype>::readListFile(const string& source) { 
    // Read the source to parse the data.
    LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
    hdf_filenames_.clear();
    labels_.clear();
    subclasses_.clear();
    videoId_.clear();
    mirrors_.clear();
    std::ifstream source_file(source.c_str());
    if (source_file.is_open()) {
      string line;
      char aux[500];
      int label;
      int gait;
      int videoId;
      int mirror;
      while (getline(source_file, line)) {
        if (line[0] != '#') { // Deal with comments.
          sscanf(line.c_str(), "%s ; %d ; %d ; %d ; %d", aux, &label, &gait, &videoId, &mirror);
          string path(aux);
          hdf_filenames_.push_back(path);
          labels_.push_back(label);
          subclasses_.push_back(gait);
          videoId_.push_back(videoId);
          mirrors_.push_back(mirror);
        }
      }
    } else {
      LOG(FATAL) << "Failed to open source file: " << source;
    }
    CHECK_GT(hdf_filenames_.size(), 0) << "Source file must contain at least 1 filename: " << source;
    source_file.close();
    num_files_ = hdf_filenames_.size();
    LOG(INFO) << "Number of HDF5 files: " << num_files_;
    CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "<< source;
  }

  template <typename Dtype>
  void FlowDataLayer<Dtype>::TransformData(Dtype* data, const float dfactor, const float meanval, const int elements) {
  	for (int i=0;i<elements;i++) {
		data[i] = (data[i] * dfactor) + (meanval * dfactor);
  	}
  }

template <typename Dtype>
  void FlowDataLayer<Dtype>::showBlob(const Blob<Dtype>* top) {
    std::ofstream myfile;
    myfile.open("/home/GAIT/blob.txt");
    vector<int> shape = top->shape();
    const Dtype* data = top->cpu_data();
    int index = 0;
    for(int i=0;i<shape[0];i++) {
      for(int j=0;j<shape[1];j++) {
        for(int k=0;k<shape[2];k++) {
          for(int l=0;l<shape[3];l++) {
            myfile << data[index] << " ";
            index++;
          }
          myfile << "\n";
        }
        myfile << "\n";
      }
      myfile << "\n";
    }
    myfile.close();
  }

template <typename Dtype>
  void FlowDataLayer<Dtype>::getBalancedPartition(const vector<string>& files, const vector<unsigned int>& subclasses, 
    const vector<int>& labels, vector<vector<bool> >& usedSamples, vector<string>& part, vector<int>& labels_part) {

    // Clear previous vector.
    part.clear();
    labels_part.clear();

    // Count samples of each kind.
    std::set<unsigned int> kindSubclassesAux(subclasses.begin(), subclasses.end());
    vector<unsigned int> kindSubclasses(kindSubclassesAux.begin(), kindSubclassesAux.end());
    vector<unsigned int> nsamples(kindSubclasses.size(), 0);
    int i = 0;
    for (vector<unsigned int>::iterator it=kindSubclasses.begin(); it!=kindSubclasses.end(); ++it) {
      nsamples[i] = std::count(subclasses.begin(), subclasses.end(), *it);
      i++;
    }

    if (usedSamples.empty()) {
      for (vector<unsigned int>::iterator it=kindSubclasses.begin(); it!=kindSubclasses.end(); ++it) {
        usedSamples.push_back(vector<bool>());
      }

      // Get balanced partition.
      int n = *std::min_element(nsamples.begin(), nsamples.end());
      for (int i=0 ; i<kindSubclasses.size() ; i++) {
        vector<bool> mask(nsamples[i], 0);
        for (unsigned int j=0 ; j<n; j++) {
          mask[j] = true;
        }

        // Get a ramdom mask.
        std::srand(time(NULL));
        std::random_shuffle(mask.begin(), mask.end());

        int pos = 0;
        bool cont = true;
        // Find elements of the same subclass and selected in the mask.
        for (unsigned int j=0; j < subclasses.size() && cont; j++) {
          if (subclasses[j] == kindSubclasses[i] && mask[pos]) {
            part.push_back(files[j]);
            labels_part.push_back(labels[j]);
            pos++;
          }else if(subclasses[j] == kindSubclasses[i] && !mask[pos]) {
            pos++;
          }

          // We have checked all samples of the subclass.
          if (pos >= nsamples[i]) {
            cont = false;
          }
        }

        // Update used positions.
        usedSamples[i] = mask;
      }
    } else {
      // Get balanced partition.
      int n = *std::min_element(nsamples.begin(), nsamples.end());
      for (int i=0 ; i<kindSubclasses.size() ; i++) {
        // Find unused positions.
        vector<bool> available(usedSamples[i].size(), false);
        vector<unsigned int> positionsAvailable;
        for (unsigned int j=0; j<usedSamples[i].size(); j++) {
          available[j] = !usedSamples[i][j];
          if (available[j]) {
            positionsAvailable.push_back(j);
          }
        }

        // Check if there are n samples.
        int nav = std::count(available.begin(), available.end(), true);
        int m;
        if (nav < n) {
          m = nav;
        } else {
          m = n;
        }

        // Get a ramdom order.
        std::srand(time(NULL));
        std::random_shuffle(positionsAvailable.begin(), positionsAvailable.end());

        vector<bool> mask2(nsamples[i], false);
        unsigned int inserted = 0;
        for (unsigned int j=0 ; j<positionsAvailable.size() && inserted < m; j++) {
          mask2[positionsAvailable[j]] = true;
          inserted++;
        }

        int pos = 0;
        bool cont = true;
        vector<unsigned int> unusedPositions;
        // Find elements of the same subclass and selected in the mask.
        for (unsigned int j=0; j < subclasses.size() && cont; j++) {
          if (subclasses[j] == kindSubclasses[i] && mask2[pos] && available[pos]) {
            part.push_back(files[j]);
            labels_part.push_back(labels[j]);
            available[pos] = false;
            pos++;
          }else if(subclasses[j] == kindSubclasses[i] && !mask2[pos]) {
            unusedPositions.push_back(pos);
            pos++;
          }

          // We have checked all samples of the subclass.
          if (pos >= nsamples[i]) {
            cont = false;
          }
        }

        if (m < n) {
          m = n - m;

          // Get a ramdom order.
          std::srand(time(NULL));
          std::random_shuffle(unusedPositions.begin(), unusedPositions.end());

          vector<bool> mask3(nsamples[i], false);
          inserted = 0;
          for (unsigned int j=0 ; j<unusedPositions.size() && inserted < m; j++) {
            mask3[unusedPositions[j]] = true;
            inserted++;
          }

          pos = 0;
          cont = true;
          // Find elements of the same subclass and selected in the mask.
          for (unsigned int j=0; j < subclasses.size() && cont; j++) {
            if (subclasses[j] == kindSubclasses[i] && mask3[pos]) {
              part.push_back(files[j]);
              labels_part.push_back(labels[j]);
              pos++;
            }else if(subclasses[j] == kindSubclasses[i] && !mask3[pos]) {
              pos++;
            }

            // We have checked all samples of the subclass.
            if (pos >= nsamples[i]) {
              cont = false;
            }
          }

          // Update used positions.
          for (unsigned int j=0; j<usedSamples[i].size(); j++) {
            usedSamples[i][j] = usedSamples[i][j] & mask3[j];
          }
        }

                // Update used positions.
        for (unsigned int j=0; j<usedSamples[i].size(); j++) {
          usedSamples[i][j] = usedSamples[i][j] | mask2[j];
        }
      }
    }
  }


#ifdef CPU_ONLY
  STUB_GPU_FORWARD(FlowDataLayer, Forward);
#endif

  INSTANTIATE_CLASS(FlowDataLayer);
  REGISTER_LAYER_CLASS(FlowData);

}  // namespace caffe
