#ifndef MODEL_H
#define MODEL_H

#include <torch/torch.h>
#include <iostream>



class PointnetImpl : public torch::nn::Module{
    public:
        PointnetImpl();
        PointnetImpl(int num_classes);
        torch::Tensor forward(torch::Tensor x);
    private:
        int num_classes;
        torch::nn::Conv1d conv_1{nullptr};
        torch::nn::Conv1d conv_2{nullptr};
        torch::nn::Conv1d conv_3{nullptr};
        torch::nn::Conv1d conv_4{nullptr};
        
        torch::nn::Linear fc_1{nullptr};
        torch::nn::Linear fc_2{nullptr};
        torch::nn::Linear last{nullptr};
        
        torch::nn::BatchNorm bn_1{nullptr};
        torch::nn::BatchNorm bn_2{nullptr};
        torch::nn::BatchNorm bn_3{nullptr};
        torch::nn::BatchNorm bn_4{nullptr};
        torch::nn::BatchNorm bn_5{nullptr};
        torch::nn::BatchNorm bn_6{nullptr};

        torch::nn::Dropout dp_1{nullptr};
        torch::nn::Dropout dp_2{nullptr};

};

TORCH_MODULE(Pointnet);

#endif



