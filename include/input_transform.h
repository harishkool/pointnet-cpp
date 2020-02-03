#ifndef INPUT_TRANSFORM_H
#define INPUT_TRANSFORM_H
#include <torch/torch.h>

class InputTransformImpl : public torch::nn::Module{
    public:
        InputTransformImpl();
        InputTransformImpl(int K);
        torch::Tensor forward(torch::Tensor x);
    private:
        int K;
        torch::nn::Conv1d conv_1{nullptr};
        torch::nn::Conv1d conv_2{nullptr};
        torch::nn::Conv1d conv_3{nullptr};
        
        torch::nn::Linear fc_1{nullptr};
        torch::nn::Linear fc_2{nullptr};
        
        torch::nn::BatchNorm bn_1{nullptr};
        torch::nn::BatchNorm bn_2{nullptr};
        torch::nn::BatchNorm bn_3{nullptr};
};

class FeatureTransformImpl : public torch::nn::Module{
    public:
        FeatureTransformImpl();
        FeatureTransformImpl(int K);
        torch::Tensor forward(torch::Tensor x);
    private:
        int K;
        torch::nn::Conv1d conv_1{nullptr};
        torch::nn::Conv1d conv_2{nullptr};
        torch::nn::Conv1d conv_3{nullptr};
        
        torch::nn::Linear fc_1{nullptr};
        torch::nn::Linear fc_2{nullptr};
        
        torch::nn::BatchNorm bn_1{nullptr};
        torch::nn::BatchNorm bn_2{nullptr};
        torch::nn::BatchNorm bn_3{nullptr};
};

TORCH_MODULE(InputTransform);
TORCH_MODULE(FeatureTransform);

#endif