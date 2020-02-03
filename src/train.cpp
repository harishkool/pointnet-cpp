#include <torch/torch.h>
#include <iostream>
#include <dirent.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>
#include <fstream>
#include "model.h"
#include "train.h"
#include <H5Cpp.h>

using namespace HighFive;

torch::Device device(torch::kCUDA);

//read pcds and return point cloud data
torch::Tensor read_pcd(std::string location){

}

// Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as type torch::Tensor
torch::Tensor read_label(std::string location){

}


std::vector<torch::Tensor> process_pcd(std::vector<std::string>pcd_list){

}

std::vector<torch::Tensor> process_labels(std::vector<int> label_list){

}

std::pair<std::vector<std::string>, std::vector<int>> get_data(std::string folder)
{   
    std::vector<std::string>inputs;
    std::vector<int>labels;
    struct dirent *entry;
    DIR *dir = opendir(folder.c_str());
    // if (dir == NULL)
    // {

    //     return ;
    // }
    // std::cout << "folder name is " << folder << std::endl;
    // while ((entry = readdir(dir)) != NULL)
    // {
    //     std::string file = entry->d_name;
    //     std::string cpt;
    //     if (file == ".")
    //     {
    //         continue;
    //     }
    //     else if ((file.find(".") == file.npos) || (file.find("..") == file.npos))
    //     {
    //         // std::cout << file.substr(file.length() - 3) << std::endl;
    //         std::string ext = file.substr(file.length() - 3);
    //         if (ext == "pcd")
    //         {
    //             cpt = folder + file;
    //             inputs.push_back(cpt);
                
    //         }
    //     }
    // }
    std::string ext = ".pcd";
    for(int i=1;i<=2048;i++){

        std::string pcd_nme =folder + std::to_string(i) + ext;
        inputs.push_back(pcd_nme);
    }
    std::string labl_fl = folder + std::string("labels.txt");
    std::string line;
    std::ifstream myfile;
    myfile.open(labl_fl);
    std::vector<int>labls;
    while(!myfile.eof()){
        getline(myfile, line);
        labls.push_back(std::stoi(line));
    }
    myfile.close();
    return std::make_pair(inputs,labls);
}

void train(Pointnet &model, torch::optim::Optimizer &optimizer, int64_t num_epochs, int64_t batch_size)
{   
    model->to(device);
    File file("train.h5", File::ReadWrite);
    DataSet dataset = file.getDataSet("data");
    std::vector<std::vector<std::vector<size_t>>> vec;
    dataset.read(vec);
    
    torch::Tensor inpt_tensor = torch::from_blob(vec.data(), {2048, 2048, 3}, torch::kByte);
    // std::string labl_fl = std::string("/home/nagaharish/Downloads/pcd_data/train/")+ std::string("labels.txt");
    std::string labl_fl = "train_labels.txt";
    std::string line;
    std::ifstream myfile (labl_fl);
    std::vector<int>labls;
    while(myfile.good()){
        getline(myfile, line);
        labls.push_back(std::stoi(line));
    }

    torch::Tensor label_tensor = torch::from_blob(labls.data(), {2048}, torch::kByte);

    for(int epoch =0;epoch<num_epochs;epoch++){
        float mse =0;
        float Acc=0;
        for(int btch=0;btch<2048;btch++){
            torch::Tensor inp = inpt_tensor[btch];
            torch::Tensor labl = label_tensor[btch];
            inp= inp.to(torch::kFloat32).to(device);
            labl = labl.to(torch::kInt64).to(device);
            inp = inp.view({1,3,2048});
            optimizer.zero_grad();
            auto output = model->forward(inp);
            auto loss = torch::nll_loss(torch::log_softmax(output, 1), labl);
            loss.backward();
            optimizer.step();
            auto acc = output.argmax(1).eq(labl).sum();
            Acc += acc.template item<float>();
            mse += loss.template item<float>();
        }
        mse = mse/float(2048);
        Acc = Acc/float(2048);
        std::cout<<"Mse is: "<<mse<<" Accuracy is: "<<Acc<<std::endl;
    }
    std::cout<<"Training done"<<std::endl;
    torch::save(model,"model_classify.pt");
}


void test(Pointnet &model)
{   
    model->to(device);
    File file("test.h5", File::ReadWrite);
    DataSet dataset = file.getDataSet("data");
    std::vector<std::vector<std::vector<size_t>>> vec;
    dataset.read(vec);

    torch::Tensor inpt_tensor = torch::from_blob(vec.data(), {420, 2048, 3}, torch::kByte);
    // std::string labl_fl = std::string("/home/nagaharish/Downloads/pcd_data/test/")+ std::string("labels.txt");
    std::string labl_fl = "test_labels.txt";
    std::string line;
    std::ifstream myfile (labl_fl);
    std::vector<int>labls;
    while(myfile.good()){
        getline(myfile, line);
        labls.push_back(std::stoi(line));
    }

    torch::Tensor label_tensor = torch::from_blob(labls.data(), {420}, torch::kByte);
    float Acc;
    float mse;
    for(int btch=0;btch<420;btch++)
    {
        torch::Tensor inp = inpt_tensor[btch];
        torch::Tensor labl = label_tensor[btch];
        inp = inp.to(torch::kFloat32).to(device);
        labl = labl.to(torch::kInt64).to(device);
        inp = inp.view({1, 3, 2048});
        auto output = model->forward(inp);
        auto loss = torch::nll_loss(torch::log_softmax(output, 1), labl);
        auto acc = output.argmax(1).eq(labl).sum();
        Acc += acc.template item<float>();
        mse += loss.template item<float>();
    }
        mse = mse/float(420);
        Acc = Acc/float(420);
        std::cout<<"Mse is: "<<mse<<" Accuracy is: "<<Acc<<std::endl;
    std::cout<<"Testing done"<<std::endl;
}


int main()
{
    // std::pair<std::vector<std::string>,std::vector<int>>train_pair = get_data("/home/nagaharish/Downloads/pcd_data/train/");
    // std::pair<std::vector<std::string>,std::vector<int>>test_pair = get_data("/home/nagaharish/Downloads/pcd_data/test/");

    // std::vector<torch::Tensor>pcds = process_pcd(train_pair.first);
    // std::vector<torch::Tensor>labels = process_labels(train_pair.second);
    Pointnet model(40);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
    train(model, optimizer, 10, 32);
    test(model);
    std::cout<<"Done"<<std::endl;
    return 0;
}