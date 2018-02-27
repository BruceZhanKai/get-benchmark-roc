// train_bayes_caffe.cpp : Defines the entry point for the console application.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "libBayesian.h"
#include "libCaffe.h"
#include "basics.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <cstdlib>
#include <cstdio>

#include <unistd.h>
#include <dirent.h>

#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	//system("pause");

	cout << "Welcome to use Startup Recognition Engine." << endl;

    FacialCaffe facialcaffe;
    FacialBayesian facialbayesian;


    if(!facialcaffe.LoadModel(argv))
    {
        std::cout << "[main] facialcaffe LoadModel fail\n";
		return false;
    }

    if(!facialbayesian.LoadModel(argv))
    {
        std::cout << "[main] facialbayesian LoadModel fail\n";
		return false;
    }


	int times = 600;
	string input_path, output_path, input_txt;
	int threshold = 0;
	double ratio;
	//cout << "Bayes threshold?" << endl;
	//cout << ">";
	//cin >> threshold;
	cout << "./xxxxxxxx/...." << endl;
	cout << "input_path>";
	cin >> input_path;
	cout << "./input_path/xxxx.txt" << endl;
	cout << "input_txt>";
	cin >> input_txt;
	cout << "how about the times in txt (600) (6000)" << endl;
	cout << "times>";
	cin >> times;

	bool show_image = 0;
	//cout << "Show the image ? No = 0 , Yes = 1 " << endl;
	//cout << "Show>";
	//cin >> show_image;

	ifstream file("./" + input_path + "/" + input_txt + ".txt");
	if (!file)
	{
		cout << "Can't open file" << endl;
		system("pause");
		return 0;
	}

    std::string ResultName="[";
    std::string cnn_str = "./Parameter.txt";
    BZ::IfstreamInfo<std::string> FR_str;
    FR_str.DoIfstreamItem(cnn_str,"path_cnn_model:");
    if(FR_str.GetBool()){ResultName+=FR_str.GetValue();ResultName+="][";};
    FR_str.DoIfstreamItem(cnn_str,"name_cnn_model:");
    if(FR_str.GetBool()){ResultName+=FR_str.GetValue();ResultName+="][";};
    FR_str.DoIfstreamItem(cnn_str,"layer:");
    if(FR_str.GetBool()){ResultName+=FR_str.GetValue();ResultName+="][";};
    FR_str.DoIfstreamItem(cnn_str,"joint:");
    if(FR_str.GetBool()){ResultName+=FR_str.GetValue();ResultName+="][";};
    FR_str.DoIfstreamItem(cnn_str,"feature_size:");
    if(FR_str.GetBool()){ResultName+=FR_str.GetValue();ResultName+="][";};
    FR_str.DoIfstreamItem(cnn_str,"feature_pca_size:");
    if(FR_str.GetBool()){ResultName+=FR_str.GetValue();ResultName+="]";};


    char path3[1000] = "Result/", path4[1000];
    mkdir(path3, 0777);

    time_t now = time(0);
	tm *ltm = localtime(&now);
    std::string result_day;
	std::stringstream p9;
	p9 << input_path << "-"<< ResultName << "-" << ltm->tm_year + 1900 << "-" << ltm->tm_mon + 1 << "-" << ltm->tm_mday;
	p9 >> result_day;
	strcpy(path4, result_day.c_str());
    strcat(path3, path4);
    mkdir(path3, 0777);

    std::string PATH(path3);

	std::string ROC_path = PATH + "/ROC.txt";
	std::ofstream ROC_file;
	ROC_file.open(ROC_path, ios::out);
	ROC_file.close();


	std::string record_path = PATH + "/record.txt";
	std::ofstream record_file;
	record_file.open(record_path, ios::out);
	record_file.close();

	string ratio_path = PATH + "/ratio.txt";
	ofstream ratio_file;
	ratio_file.open(ratio_path, ios::out);
	ratio_file.close();

	string thres_path = PATH + "/thres.txt";
	ofstream thres_file;
	thres_file.open(thres_path, ios::out);
	thres_file.close();


	string feature1_path = PATH + "/feature1.txt";
	ofstream feature1_file;
	feature1_file.open(feature1_path, ios::out);
	feature1_file.close();

	string feature2_path = PATH + "/feature2.txt";
	ofstream feature2_file;
	feature2_file.open(feature2_path, ios::out);
	feature2_file.close();


	string acc_path = PATH + "/ACC.txt";
	ofstream acc_file;
	acc_file.open(acc_path, ios::out);
	acc_file.close();

	int name_size = 4;

	vector<bool> truth_is_true;
	vector<double> ratio_all;

	for (int k = 0; k < times; k++)
	{
		string line[8];
		//bool Result_Is_True = 0;
		bool Truth_Is_True = 1;
		file >> line[0];
		file >> line[1];
		file >> line[2];
		//cout << "line[0]=" << line[0] << endl;
		//cout << "line[1]=" << line[1] << endl;
		//cout << "line[2]=" << line[2] << endl;

		if (line[1].length() == 1)
			line[3] = "./" + input_path + "/" + line[0] + "/" + line[0] + "_000" + line[1] + ".jpg";
		else if (line[1].length() == 2)
			line[3] = "./" + input_path + "/" + line[0] + "/" + line[0] + "_00" + line[1] + ".jpg";
		else if (line[1].length() == 3)
			line[3] = "./" + input_path + "/" + line[0] + "/" + line[0] + "_0" + line[1] + ".jpg";


		if (line[2].length() == 1)
			line[4] = "./" + input_path + "/" + line[0] + "/" + line[0] + "_000" + line[2] + ".jpg";
		else if (line[2].length() == 2)
			line[4] = "./" + input_path + "/" + line[0] + "/" + line[0] + "_00" + line[2] + ".jpg";
		else if (line[2].length() == 3)
			line[4] = "./" + input_path + "/" + line[0] + "/" + line[0] + "_0" + line[2] + ".jpg";
		else if (line[2].length() >= name_size)
		{
			Truth_Is_True = 0;
			file >> line[7];
			if (line[7].length() == 1)
				line[4] = "./" + input_path + "/" + line[2] + "/" + line[2] + "_000" + line[7] + ".jpg";
			else if (line[7].length() == 2)
				line[4] = "./" + input_path + "/" + line[2] + "/" + line[2] + "_00" + line[7] + ".jpg";
			else if (line[7].length() == 3)
				line[4] = "./" + input_path + "/" + line[2] + "/" + line[2] + "_0" + line[7] + ".jpg";
		}

		cv::Mat image1 = imread(line[3], 1);
		cv::Mat image2 = imread(line[4], 1);
		if (image1.empty() || image2.empty())
		{
			if (image1.empty())
			{
				cout << "image1=empty,path=" << line[3] << endl;
			}
			if (image2.empty())
			{
				cout << "image2=empty,path=" << line[4] << endl;
			}
			system("pause");
			return 0;
		}
        std::vector<float> feature_vector1 = facialcaffe.Predict(image1);
        std::vector<float> feature_vector2 = facialcaffe.Predict(image2);

        int SIZE1=feature_vector1.size();

		for (int i = 0; i < SIZE1; i++)
		{
			//if (i < 2)
			//{
			//	cout << "data1[i]=" << data1[i] << endl;
			//	cout << "data2[i]=" << data2[i] << endl;
			//}

			if (i == SIZE1 - 1)
			{
				feature1_file.open(feature1_path, ios::app);
				feature1_file << feature_vector1[i] << "\n";
				feature1_file.close();
				feature2_file.open(feature2_path, ios::app);
				feature2_file << feature_vector2[i] << "\n";
				feature2_file.close();
			}
			else
			{
				feature1_file.open(feature1_path, ios::app);
				feature1_file << feature_vector1[i] << "\t";
				feature1_file.close();
				feature2_file.open(feature2_path, ios::app);
				feature2_file << feature_vector2[i] << "\t";
				feature2_file.close();
			}
		}

        ratio = facialbayesian.Verify(feature_vector1, feature_vector2);
		//cout << "ratio=" << ratio << endl;

		if (show_image)
		{
			cv::imshow("image1", image1);
			cv::imshow("image2", image2);
			if (Truth_Is_True)
				cout << "positive" << endl;
			else
				cout << "negative" << endl;
			waitKey(1500);
			system("pause");
		}


		truth_is_true.push_back(Truth_Is_True);
		ratio_all.push_back(ratio);

		ratio_file.open(ratio_path, ios::app);
		//ratio_file.setf(ios::fixed, ios::floatfield);
		//ratio_file.precision(10);
		ratio_file << Truth_Is_True << "\t" << ratio << endl;
		ratio_file.close();

		//if (Result_Is_True && Truth_Is_True)
		//{
		//	TP[0]++;
		//}
		//else if (Result_Is_True && !Truth_Is_True)
		//{
		//	FP[0]++;
		//}
		//else if (!Result_Is_True && !Truth_Is_True)
		//{
		//	TN[0]++;
		//}
		//else if (!Result_Is_True && Truth_Is_True)
		//{
		//	FN[0]++;
		//}
		//total[0]++;


		//total[10] = { 0 },
		//	TPR[10] = { 0 }, TP[10] = { 0 }, FN[10] = { 0 },
		//	FPR[10] = { 0 }, FP[10] = { 0 }, TN[10] = { 0 };


	}

	//float ratio_final[600];
	//bool positive[600];

	vector<float> ACC;

	for (int i = 0; i < 10; i++)
	{
		string ii;
		stringstream iii;
		iii << i;
		iii >> ii;
		iii.str("");
		iii.clear();
		vector<float> acc;
		vector<int> thres;
		vector<bool> Positive;
		vector<float> Ratio;

		for (int j = 0; j < 600; j++)
		{
			Ratio.push_back(ratio_all[i * 600 + j]);
			Positive.push_back(truth_is_true[i * 600 + j]);
		}

		float max_thres = *max_element(Ratio.begin(), Ratio.end()) + 1;
		float min_thres = *min_element(Ratio.begin(), Ratio.end()) - 1;

		float gap = abs(max_thres - min_thres) / 6000.0f;

		for (float k = max_thres; k > min_thres; k -= gap)
		{
			float total[10] = { 0 },
				TPR[10] = { 0 }, TP[10] = { 0 }, FN[10] = { 0 },
				FPR[10] = { 0 }, FP[10] = { 0 }, TN[10] = { 0 };


			for (int j = 0; j < Ratio.size(); j++)
			{

				bool Result_Is_True;

				if (Ratio[j] > k)
					Result_Is_True = 1;
				else
					Result_Is_True = 0;


				//============positive==================
				if (Result_Is_True && Positive[j])
				{
					TP[0]++;
				}
				else if (!Result_Is_True && Positive[j])
				{
					FN[0]++;
				}
				//======================================

				//============nagetive==================
				else if (Result_Is_True && !Positive[j])
				{
					FP[0]++;
				}
				else if (!Result_Is_True && !Positive[j])
				{
					TN[0]++;
				}
				//======================================
				total[0]++;



			}
			thres.push_back(k);
			//cout << "thres=" << k << "\n" << endl;

			acc.push_back((TP[0] + ((FP[0] + TN[0]) - FP[0])) / ((TP[0] + FN[0]) + (FP[0] + TN[0])));
			//cout << "acc=" << (TP[0] + ((FP[0] + TN[0]) - FP[0])) / ((TP[0] + FN[0]) + (FP[0] + TN[0])) << "\n" << endl;

		}

		ofstream file_save;
		file_save.open("./0_data/acc_" + ii + ".txt", ios::out);
		for (int k = 0; k < acc.size(); k++)
			file_save << k << "\t" << thres[k] << "\t" << acc[k] << endl;
		file_save.close();
		float max = *max_element(acc.begin(), acc.end());
		ACC.push_back(max);


		acc_file.open(acc_path, ios::app);
		acc_file << "max[" << i << "]=" << max << endl;
		acc_file.close();
		cout << "max[" << i << "]=" << max << endl;
		system("pause");
	}


	float sum = 0;
	for (int i = 0; i < ACC.size(); i++)
		sum += ACC[i];

	sum = sum / ACC.size();
	cout << "acc_averg=" << sum << endl;

	float std = sqrt((pow(ACC[0] - sum, 2) + pow(ACC[1] - sum, 2) + pow(ACC[2] - sum, 2) + pow(ACC[3] - sum, 2) + pow(ACC[4] - sum, 2) + pow(ACC[5] - sum, 2) + pow(ACC[6] - sum, 2) + pow(ACC[7] - sum, 2) + pow(ACC[8] - sum, 2) + pow(ACC[9] - sum, 2)) / 9);
	cout << "std=" << std << endl;

	float std_em = std / sqrt(10);
	cout << "std_em=" << std_em << endl;

	acc_file.open(acc_path, ios::app);
	acc_file << "acc_averg=" << sum << endl;
	acc_file << "std=" << std << endl;
	acc_file << "std_em=" << std_em << endl;
	acc_file.close();

	cout << "ACC Complete." << endl;
	system("pause");






	record_file.open(record_path, ios::app);
	record_file << "truth_is_true size = " << truth_is_true.size() << endl;
	record_file << "ratio_all size = " << ratio_all.size() << endl;
	record_file.close();

	if (truth_is_true.size() != ratio_all.size())
	{
		cout << "they should be the same size" << endl;
		system("pause");
		return 0;
	}

	float max_thres = *max_element(ratio_all.begin(), ratio_all.end()) + 1;
	float min_thres = *min_element(ratio_all.begin(), ratio_all.end()) - 1;

	float gap = abs(max_thres - min_thres) / 6000.0f;


	record_file.open(record_path, ios::app);
	record_file << "max_threshold = " << max_thres << endl;
	record_file << "min_threshold = " << min_thres << endl;
	record_file.close();


	for (float i = max_thres; i > min_thres; i -= gap)
	{
		float total[10] = { 0 },
			TPR[10] = { 0 }, TP[10] = { 0 }, FN[10] = { 0 },
			FPR[10] = { 0 }, FP[10] = { 0 }, TN[10] = { 0 };


		for (int j = 0; j < ratio_all.size(); j++)
		{

			bool Result_Is_True;

			if (ratio_all[j] > i)
				Result_Is_True = 1;
			else
				Result_Is_True = 0;


			//============positive==================
			if (Result_Is_True && truth_is_true[j])
			{
				TP[0]++;
			}
			else if (!Result_Is_True && truth_is_true[j])
			{
				FN[0]++;
			}
			//======================================

			//============nagetive==================
			else if (Result_Is_True && !truth_is_true[j])
			{
				FP[0]++;
			}
			else if (!Result_Is_True && !truth_is_true[j])
			{
				TN[0]++;
			}
			//======================================
			total[0]++;



		}

		TPR[0] = TP[0] / (TP[0] + FN[0]);
		FPR[0] = FP[0] / (FP[0] + TN[0]);

		cout << "threshold=" << i << endl;
		cout << "TPR=" << TP[0] << "/" << (TP[0] + FN[0]) << "=" << TPR[0] << endl;
		cout << "FPR=" << FP[0] << "/" << (FP[0] + TN[0]) << "=" << FPR[0] << endl;

		thres_file.open(thres_path, ios::app);
		thres_file << "threshold = " << i << endl;
		thres_file << "TPR=" << TPR[0] << "\t" << TP[0] << "\t" << (TP[0] + FN[0]) << "\nFPR=" << FPR[0] << "\t" << FP[0] << "\t" << (FP[0] + TN[0]) << endl;
		thres_file.close();

		ROC_file.open(ROC_path, ios::app);
		ROC_file.setf(ios::fixed, ios::floatfield);
		ROC_file.precision(6);
		ROC_file << TPR[0] << "\t" << FPR[0] << endl;
		ROC_file.close();



	}




	cout << "ROC complete" << endl;
	system("pause");
	return 0;
}



/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
//#include <caffe/caffe.hpp>
#include "features.h"
using namespace std;
using namespace cv;
//using namespace caffe;

int main(int argc, char **argv)
{
cout << "Welcome to use XYZ Robot Recognition Engine." << endl;
cout << "Initialize..." << endl;

//if (argc != 5)
//if (argc != 9)
if (argc != 1)
{
std::cerr << "Usage: " << argv[0]
<< " deploy.prototxt network.caffemodel"
<< " mean.binaryproto labels.txt " << std::endl;
return 1;
}
::google::InitGoogleLogging(argv[0]);
//string model_file   = argv[1];         //deploy_gender.prototxt
//string trained_file = argv[2];         //gender_net.caffemodel
//string mean_file    = argv[3];         //mean.binaryproto
//string label_file   = argv[4];         //labels.txt
string model_file = "./model1/VGG_FACE_deploy.prototxt";         //deploy_gender.prototxt
cout << "model_file setting down..." << endl;

string trained_file = "./model1/VGG_FACE.caffemodel";         //gender_net.caffemodel
cout << "weights_file setting down..." << endl;

string mean_file = "./model1/mean.binaryproto";         //mean.binaryproto
cout << "mean_file setting down..." << endl;

string label_file = "./model1/names.txt";         //labels.txt
cout << "label_file setting down..." << endl;

Classifier classifier(model_file, trained_file, mean_file, label_file);

cout << "DNN loading... over" << endl;
//======================================================

const int SIZE = 4096;

Mat img = imread("ak.png", -1);
Mat dstMat;
resize(img, dstMat, Size(224, 224), 0, 0, CV_INTER_LINEAR);
cout << "extracting feature" << endl;
const float* feature_vector = classifier.Classify(dstMat);
cout << "feature extracting finished" << endl;
//for (int d=0; d<SIZE; d++){
//cout<<"feature_vector["<<d<<"]="<<feature_vector[d]<<endl;
//}
cout << "ouput to txt" << endl;
ofstream fout("test1.txt");
if (!fout) {
cout << "無法寫入檔案\n";
return 1;
}

for (int i = 0; i<SIZE; i++){
fout << feature_vector[i] << "\t";
}
fout << endl;
cout << "finish output" << endl;
return 0;
}


*/




/*

// lfw_get_6000.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <fstream>
#include <sstream>
//===================================================================
//OpenCV2.4.11
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <stdio.h>
#include <vector>
#include <string>
#include <time.h>
#include <iomanip>
#include <windows.h>                                                           //與使用此函數QueryPerformanceCounter有關,若移動至Linux須拿掉
#include <io.h>
#include <direct.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <stdlib.h>
//===================================================================
#include <math.h>
//===================================================================

using namespace std;
using namespace cv;


int main(int argc, char **argv)
{
	int name_size = 4;
	string input_path, output_path;
	cout << "input floder name?" << endl;
	cout << "check any word isn't wrong" << endl;
	cout << ">";
	cin >> input_path;
	cout << "output floder name?" << endl;
	cout << ">";
	cin >> output_path;

	string make_path[3];
	make_path[0] = "C:/feature_collect/" + output_path;
	char cahr_path0[200];
	strcpy(cahr_path0, make_path[0].c_str());
	_mkdir(cahr_path0);


	string error_path = "C:/feature_collect/" + output_path + "/error.txt";
	ofstream file_error;
	file_error.open(error_path, ios::out);
	file_error.close();


	int times = 6000;
	ifstream file("c:/feature_collect/" + input_path + "/pairs.txt");
	if (!file)
	{
		cout << "Can't open file" << endl;
		system("pause");
		return 0;
	}
	for (int i = 0; i < times; i++)
	{
		string line[8];

		file >> line[0];
		file >> line[1];
		file >> line[2];

		make_path[1] = "C:/feature_collect/" + output_path + "/" + line[0];
		char cahr_path1[200];
		strcpy(cahr_path1, make_path[1].c_str());
		_mkdir(cahr_path1);


		if (line[1].length() == 1)
		{
			line[3] = "C:/feature_collect/" + input_path + "/" + line[0] + "/" + line[0] + "_000" + line[1] + ".jpg";
			line[5] = "C:/feature_collect/" + output_path + "/" + line[0] + "/" + line[0] + "_000" + line[1] + ".jpg";
		}
		else if (line[1].length() == 2)
		{
			line[3] = "C:/feature_collect/" + input_path + "/" + line[0] + "/" + line[0] + "_00" + line[1] + ".jpg";
			line[5] = "C:/feature_collect/" + output_path + "/" + line[0] + "/" + line[0] + "_00" + line[1] + ".jpg";
		}
		else if (line[1].length() == 3)
		{
			line[3] = "C:/feature_collect/" + input_path + "/" + line[0] + "/" + line[0] + "_0" + line[1] + ".jpg";
			line[5] = "C:/feature_collect/" + output_path + "/" + line[0] + "/" + line[0] + "_0" + line[1] + ".jpg";
		}



		if (line[2].length() == 1)
		{
			line[4] = "C:/feature_collect/" + input_path + "/" + line[0] + "/" + line[0] + "_000" + line[2] + ".jpg";
			line[6] = "C:/feature_collect/" + output_path + "/" + line[0] + "/" + line[0] + "_000" + line[2] + ".jpg";
		}
		else if (line[2].length() == 2)
		{
			line[4] = "C:/feature_collect/" + input_path + "/" + line[0] + "/" + line[0] + "_00" + line[2] + ".jpg";
			line[6] = "C:/feature_collect/" + output_path + "/" + line[0] + "/" + line[0] + "_00" + line[2] + ".jpg";
		}
		else if (line[2].length() == 3)
		{
			line[4] = "C:/feature_collect/" + input_path + "/" + line[0] + "/" + line[0] + "_0" + line[2] + ".jpg";
			line[6] = "C:/feature_collect/" + output_path + "/" + line[0] + "/" + line[0] + "_0" + line[2] + ".jpg";

		}
		else if (line[2].length() >= name_size)
		{
			make_path[2] = "C:/feature_collect/" + output_path + "/" + line[2];
			char cahr_path2[200];
			strcpy(cahr_path2, make_path[2].c_str());
			_mkdir(cahr_path2);

			file >> line[7];
			if (line[7].length() == 1)
			{
				line[4] = "C:/feature_collect/" + input_path + "/" + line[2] + "/" + line[2] + "_000" + line[7] + ".jpg";
				line[6] = "C:/feature_collect/" + output_path + "/" + line[2] + "/" + line[2] + "_000" + line[7] + ".jpg";
			}
			else if (line[7].length() == 2)
			{
				line[4] = "C:/feature_collect/" + input_path + "/" + line[2] + "/" + line[2] + "_00" + line[7] + ".jpg";
				line[6] = "C:/feature_collect/" + output_path + "/" + line[2] + "/" + line[2] + "_00" + line[7] + ".jpg";
			}
			else if (line[7].length() == 3)
			{
				line[4] = "C:/feature_collect/" + input_path + "/" + line[2] + "/" + line[2] + "_0" + line[7] + ".jpg";
				line[6] = "C:/feature_collect/" + output_path + "/" + line[2] + "/" + line[2] + "_0" + line[7] + ".jpg";
			}
		}


		Mat image1 = imread(line[3], 1);
		Mat image2 = imread(line[4], 1);

		if (image1.empty())
		{
			file_error.open(error_path, ios::app);
			file_error << "image1=" << line[0] << "\t" << line[1] << "\tfail" << endl;
			file_error.close();
			//cout << line[0] << "\t" << line[1] << "\tfail" << endl;
		}
		else
		{
			imwrite(line[5], image1);
		}



		if (image2.empty())
		{
			if (line[2].length() >= name_size)
			{
				file_error.open(error_path, ios::app);
				file_error << "image2=" << line[2] << "\t" << line[7] << "\tfail" << endl;
				file_error.close();
			}
			else
			{
				file_error.open(error_path, ios::app);
				file_error << "image2=" << line[0] << "\t" << line[2] << "\tfail" << endl;
				file_error.close();
			}
			//cout << line[0] << "\t" << line[2] << "\tfail" << endl;
		}
		else
		{
			imwrite(line[6], image2);
		}

		if (line[2].length() <= name_size)
			cout << line[0] << "\t" << line[1] << " \t" << line[2] << endl;
		else
			cout << line[0] << "\t" << line[1] << " \t" << line[2] << " \t" << line[7] << endl;


	}





	return 0;
}




*/
