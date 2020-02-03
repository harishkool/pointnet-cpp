#ifndef TRAIN_H
#define TRAIN_H

#include <torch/torch.h>
#include <torch/script.h>
#include <vector>

//read pcds and return point cloud data
torch::Tensor read_pcd(std::string location);

// Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as type torch::Tensor
torch::Tensor read_label(std::string location);


std::vector<torch::Tensor> process_pcd(std::vector<std::string>pcd_list);

std::vector<torch::Tensor> process_labels(std::vector<int> label_list);


std::pair<std::vector<std::string>, std::vector<int>> get_data(std::string folder);

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    /* data */
    // Should be 2 tensors
    std::vector<torch::Tensor> states, labels;
    size_t ds_size;
public:
    CustomDataset(std::vector<std::string> list_images, std::vector<int> list_labels) {
        states =  process_pcd(list_images);
        labels = process_labels(list_labels);
        ds_size = states.size();
    };
    
    torch::data::Example<> get(size_t index) override {
        /* This should return {torch::Tensor, torch::Tensor} */
        torch::Tensor sample_img = states.at(index);
        torch::Tensor sample_label = labels.at(index);
        return {sample_img.clone(), sample_label.clone()};
    };
    
    torch::optional<size_t> size() const override {
        return ds_size;
    };
};


#endif