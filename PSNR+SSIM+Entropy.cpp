#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
using namespace cv;
double psnr, ssim,Q1,Q2,q1;
void get_psnr(const Mat& I1, const Mat& I2);
void get_ssim(const Mat& I1, const Mat& I2);


void get_psnr(const Mat& I1, const Mat& I2){
	Mat s1;
	absdiff(I1, I2, s1);       //  in opencv we use AbsDiff to calculate the absolute value of two arrays
	s1.convertTo(s1, CV_32F);  // 8-digit usigned char cannot be squared
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);         //sum of each channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) 
		psnr = 0;
	else
	{
		double  mse = sse / (double)(I1.channels() * I1.total());//MSE
		double psnr = 10.0*log10((255 * 255) / mse);
		cout << "PSNR=" << psnr << endl;
		cout << "\n" << endl;
	}
}

void get_ssim(const Mat& I1, const Mat& I2){
	
	const double C1 = 6.5025, C2 = 58.5225;
	Mat I2_2 = I2.mul(I2);
	Mat I1_2 = I1.mul(I1);
	Mat I1_I2 = I1.mul(I2);
	Mat mu1, mu2;
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);
	Mat sigma1_2, sigma2_2, sigma12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;
	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);
	Mat ssim_map;
	divide(t3, t1, ssim_map);
	Scalar mssim = mean(ssim_map);
	ssim = (mssim(0) + mssim(1) + mssim(2)) / 3;
	cout << "SSIM= " << ssim << endl;
}

double Entropy(Mat img)
{
	
	double temp[256];
	for (int i = 0; i<256; i++)
	{
		temp[i] = 0.0;
	}
	for (int m = 0; m<img.rows; m++)
	{
		const uchar* t = img.ptr<uchar>(m);
		for (int n = 0; n<img.cols; n++)
		{
			int i = t[n];
			temp[i] = temp[i] + 1;
		}
	}

	for (int i = 0; i<256; i++)
	{
		temp[i] = temp[i] / (img.rows*img.cols);
	}

	double result = 0;
	for (int i = 0; i<256; i++)
	{
		if (temp[i] == 0.0)
			result = result;
		else
			result = (result - temp[i] * (log(temp[i]) / log(2.0)))/8;
	}
	return result;
}



void main()
{
	Mat I1 = imread("hazy.jpg",1);
	Mat I2 = imread("haze-free.jpg", 1);
	Mat I11,I22;
	cvtColor(I1, I11, CV_BGR2GRAY);
	cvtColor(I2, I22, CV_BGR2GRAY);
	get_psnr(I1, I2);
	I1.convertTo(I1, CV_32F);
	I2.convertTo(I2, CV_32F);
	get_ssim(I1, I2);
	Q1 = Entropy(I11);
	q1 = Entropy(I22);
	cout << "Entropy of hazy£º " << Q1 <<"\n"<< endl;
	cout << "Entropy of haze-free£º " << q1 << "\n" << endl;
    system("pause");

}