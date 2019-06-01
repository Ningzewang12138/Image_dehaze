/**** Written By Ningze Wang****/


#include<iostream>
#include<vector>
#include<algorithm>

using namespace std;

#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
using namespace cv;

void AirlightEstimation(const Mat&);
void guidedeFilter(Mat&, Mat&, Mat&, int, float);
void makeDepth32f(Mat&, Mat&);
void est_mean(const Mat&);

double A[3];
int AX = 0, AY = 0;
Point a;
double im_mean, A_mean;
void AirlightEstimation(const Mat& image) {
	int nMinDistance = 65536;
	int nDistance;
	int nX, nY;

	int nMaxIndex;
	double dpScore[3];
	double dpMean[3];
	double dpStds[3];

	Mat mat_mean, mat_dev;

	float afMean[4] = { 0 };
	float afScore[4] = { 0 };
	float nMaxScore = 0;

	int nWid = image.cols;
	int nHei = image.rows;

	vector<Mat> channels;

	Mat UpperLeft(nWid / 2, nHei / 2, CV_8UC3);
	Mat UpperRight(nWid / 2, nHei / 2, CV_8UC3);
	Mat DownLeft(nWid / 2, nHei / 2, CV_8UC3);
	Mat DownRight(nWid / 2, nHei / 2, CV_8UC3);

	Mat ImB(nWid / 2, nHei / 2, CV_8UC1);
	Mat ImG(nWid / 2, nHei / 2, CV_8UC1);
	Mat ImR(nWid / 2, nHei / 2, CV_8UC1);

	UpperLeft = image(Range(0, nHei / 2), Range(0, nWid / 2));
	UpperRight = image(Range(0, nHei / 2), Range(nWid / 2, nWid));
	DownLeft = image(Range(nHei / 2, nHei), Range(0, nWid / 2));
	DownRight = image(Range(nHei / 2, nHei), Range(nWid / 2, nWid));

	//compare to the threashold and decide if the circle is over,if is bigger than divide the block
	if (nWid*nHei > 200)
	{
		//UpperLeft block
		split(UpperLeft, channels);
		ImB = channels.at(0);
		ImG = channels.at(1);
		ImR = channels.at(2);
		meanStdDev(ImR, mat_mean, mat_dev);
		dpMean[0] = mat_mean.at<double>(0, 0);
		dpStds[0] = mat_dev.at<double>(0, 0);
		meanStdDev(ImG, mat_mean, mat_dev);
		dpMean[1] = mat_mean.at<double>(0, 0);
		dpStds[1] = mat_dev.at<double>(0, 0);
		meanStdDev(ImB, mat_mean, mat_dev);
		dpMean[2] = mat_mean.at<double>(0, 0);
		dpStds[2] = mat_dev.at<double>(0, 0);
		dpScore[0] = dpMean[0] - dpStds[0];
		dpScore[1] = dpMean[1] - dpStds[1];
		dpScore[2] = dpMean[2] - dpStds[2];
		afScore[0] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);
		nMaxScore = afScore[0];
		nMaxIndex = 0;

		//UpperRight block
		split(UpperRight, channels);
		ImB = channels.at(0);
		ImG = channels.at(1);
		ImR = channels.at(2);
		meanStdDev(ImR, mat_mean, mat_dev);
		dpMean[0] = mat_mean.at<double>(0, 0);
		dpStds[0] = mat_dev.at<double>(0, 0);
		meanStdDev(ImG, mat_mean, mat_dev);
		dpMean[1] = mat_mean.at<double>(0, 0);
		dpStds[1] = mat_dev.at<double>(0, 0);
		meanStdDev(ImB, mat_mean, mat_dev);
		dpMean[2] = mat_mean.at<double>(0, 0);
		dpStds[2] = mat_dev.at<double>(0, 0);
		dpScore[0] = dpMean[0] - dpStds[0];
		dpScore[1] = dpMean[1] - dpStds[1];
		dpScore[2] = dpMean[2] - dpStds[2];
		afScore[1] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);
		if (afScore[1] > nMaxScore)
		{
			nMaxScore = afScore[1];
			nMaxIndex = 1;
		}

		//DownLeft block
		split(DownLeft, channels);
		ImB = channels.at(0);
		ImG = channels.at(1);
		ImR = channels.at(2);
		meanStdDev(ImR, mat_mean, mat_dev);
		dpMean[0] = mat_mean.at<double>(0, 0);
		dpStds[0] = mat_dev.at<double>(0, 0);
		meanStdDev(ImG, mat_mean, mat_dev);
		dpMean[1] = mat_mean.at<double>(0, 0);
		dpStds[1] = mat_dev.at<double>(0, 0);
		meanStdDev(ImB, mat_mean, mat_dev);
		dpMean[2] = mat_mean.at<double>(0, 0);
		dpStds[2] = mat_dev.at<double>(0, 0);
		dpScore[0] = dpMean[0] - dpStds[0];
		dpScore[1] = dpMean[1] - dpStds[1];
		dpScore[2] = dpMean[2] - dpStds[2];
		afScore[2] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);
		if (afScore[2] > nMaxScore)
		{
			nMaxScore = afScore[2];
			nMaxIndex = 2;
		}

		//DownRight block
		split(DownRight, channels);
		ImB = channels.at(0);
		ImG = channels.at(1);
		ImR = channels.at(2);
		meanStdDev(ImR, mat_mean, mat_dev);
		dpMean[0] = mat_mean.at<double>(0, 0);
		dpStds[0] = mat_dev.at<double>(0, 0);
		meanStdDev(ImG, mat_mean, mat_dev);
		dpMean[1] = mat_mean.at<double>(0, 0);
		dpStds[1] = mat_dev.at<double>(0, 0);
		meanStdDev(ImB, mat_mean, mat_dev);
		dpMean[2] = mat_mean.at<double>(0, 0);
		dpStds[2] = mat_dev.at<double>(0, 0);
		dpScore[0] = dpMean[0] - dpStds[0];
		dpScore[1] = dpMean[1] - dpStds[1];
		dpScore[2] = dpMean[2] - dpStds[2];
		afScore[3] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);
		if (afScore[3] > nMaxScore)
		{
			nMaxScore = afScore[3];
			nMaxIndex = 3;
		}
		// select subblock to go to the next circle
		switch (nMaxIndex)
		{
		case 0:
			AirlightEstimation(UpperLeft); break;
		case 1: {   AX += nWid / 2;
			AirlightEstimation(UpperRight); break; }
		case 2: { AY += nHei / 2;
			AirlightEstimation(DownLeft); break; }
		case 3: { AX += nWid / 2; AY += nHei / 2;
			AirlightEstimation(DownRight); break; }

		}
	}
	else
	{
		//select the atmospheric light value in the sub-block
		split(image, channels);
		ImB = channels.at(0);
		ImG = channels.at(1);
		ImR = channels.at(2);
		int devR, devG, devB;
		for (nY = 0; nY < nHei; nY++)
		{
			uchar *data0 = ImB.ptr<uchar>(nY);
			uchar *data1 = ImG.ptr<uchar>(nY);
			uchar *data2 = ImR.ptr<uchar>(nY);
			for (nX = 0; nX < nWid; nX++)
			{
				//RGB color space distance
				devR = float(255 - data2[nX]);
				devG = float(255 - data1[nX]);
				devB = float(255 - data0[nX]);
				nDistance = int(sqrt(devR*devR + devG*devG + devB*devB));
				if (nMinDistance>nDistance)
				{
					nMinDistance = nDistance;
					a.x = nX;
					a.y = nY;
					A[0] = data0[nX];
					A[1] = data1[nX];
					A[2] = data2[nX];
				}
			}
		}
	}
}

void guidedFilter(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon)
{
	CV_Assert(radius >= 2 && epsilon > 0);
	CV_Assert(source.data != NULL && source.channels() == 1);
	CV_Assert(guided_image.channels() == 1);
	CV_Assert(source.rows == guided_image.rows && source.cols == guided_image.cols);

	Mat guided;
	if (guided_image.data == source.data)
	{
		//make a copy  
		guided_image.copyTo(guided);
	}
	else
	{
		guided = guided_image;
	}
	//extend input to 32-digit float  
	Mat source_32f, guided_32f;
	makeDepth32f(source, source_32f);
	makeDepth32f(guided, guided_32f);

	//I*p,I*I  
	Mat mat_Ip, mat_I2;
	multiply(guided_32f, source_32f, mat_Ip);
	multiply(guided_32f, guided_32f, mat_I2);

	//averages  
	Mat mean_p, mean_I, mean_Ip, mean_I2;
	Size win_size(2 * radius + 1, 2 * radius + 1);
	boxFilter(source_32f, mean_p, CV_32F, win_size);
	boxFilter(guided_32f, mean_I, CV_32F, win_size);
	boxFilter(mat_Ip, mean_Ip, CV_32F, win_size);
	boxFilter(mat_I2, mean_I2, CV_32F, win_size);
	//covariance of Ip and variance of I
	//D(x)=E(X2)-(EX)2
	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
	Mat var_I = mean_I2 - mean_I.mul(mean_I);
	var_I += epsilon;

	//ab  
	Mat a, b;
	divide(cov_Ip, var_I, a);
	b = mean_p - a.mul(mean_I);

	//do average on every a and b which include pixel i  
	Mat mean_a, mean_b;
	boxFilter(a, mean_a, CV_32F, win_size);
	boxFilter(b, mean_b, CV_32F, win_size);

	//(depth == CV_32F)  
	output = mean_a.mul(guided_32f) + mean_b;
}

void makeDepth32f(Mat& source, Mat& output)
{
	if (source.depth() != CV_32F)
		source.convertTo(output, CV_32F);
	else
		output = source;
}

void est_mean(const Mat& image)
{
	Scalar a = mean(image);
	double dMean[3];
	for (int k = 0; k < 3; k++)
	{
		dMean[k] = a[k];
	}
	im_mean = (dMean[0] + dMean[1] + dMean[2]) / 3;
}




int main(int argc, char*argv[])
{
	double timestart = (double)getTickCount();
	bool Y;
	Mat image = imread("D:/image/canyon2.png", 2 | 4);
	Mat image1 = imread("D:/image/canyon2.png", 0);
	cout << "Does the picture have shrot-distance sky? " << "\n" << "Yes--1£¬No--0" << endl;
	cin >> Y;
	Mat dst;
	double alpha = 1, beta = 0;
	Mat tmpimage, image2;
	tmpimage = image.clone();
	pyrDown(tmpimage, image2, Size(tmpimage.cols / 2, tmpimage.rows / 2));
	//CV_Assert do judgement on expression, true keep going, fasle then stop
	CV_Assert(!image.empty() && image.channels() == 3);
	//unify pic
	est_mean(image);
	cout << "image_mean= " << im_mean << endl;
	im_mean = im_mean / 255;
	Mat fImage, fImage1;
	image.convertTo(fImage, CV_32FC3, 1.0 / 255, 0);
	image1.convertTo(fImage1, CV_32FC1, 1.0 / 255.0);
	//fix the patch, and patch is odd number
	int hPatch = 15;
	int vPatch = 15;
	//add border for unified pic
	Mat fImageBorder;
	copyMakeBorder(fImage, fImageBorder, vPatch / 2, vPatch / 2, hPatch / 2, hPatch / 2, BORDER_REPLICATE);
	//split the channel
	//fImageBorderVector' type is  vector<mat>£¬
	vector<Mat> fImageBorderVector(3);
	split(fImageBorder, fImageBorderVector);
	//create darkChannel
	Mat darkChannel(image.rows, image.cols, CV_32FC1);
	double minTemp, minPixel;

//the minimum of the 3 channels in some area
	for (unsigned int r = 0; r < darkChannel.rows; r++)
	{
	for (unsigned int c = 0; c < darkChannel.cols; c++)
	{
	minPixel = 1.0;
	for (vector<Mat>::iterator it = fImageBorderVector.begin(); it != fImageBorderVector.end(); it++)
	{
	Mat roi(*it, Rect(c, r, hPatch, vPatch));
	minMaxLoc(roi, &minTemp);
	minPixel = min(minPixel, minTemp);
	}
	darkChannel.at<float>(r, c) = float(minPixel);
	}
	}
	namedWindow("darkChannel", 1);
	imshow("darkChannel", darkChannel);
	Mat darkChannel8U;
	darkChannel.convertTo(darkChannel8U, CV_8UC1, 255, 0);
	imwrite("darkChannel.jpg", darkChannel8U);
	

	//use quadtree algorithm get A
	AirlightEstimation(image);
	cout << "A_mean= " << (A[0] + A[1] + A[2]) / 3 << endl;
	A[0] = A[0] / 255;
	A[1] = A[1] / 255;
	A[2] = A[2] / 255;
	A_mean = (A[0] + A[1] + A[2]) / 3;
	a.x += AX; a.y += AY;
	circle(image, a, 6, Scalar(255, 0, 0), -1);
	imshow(" Point A", image);
	imwrite("point.jpg", image);

	vector<Mat>::iterator it = fImageBorderVector.begin();
	//will use the matrix in the steps to get t(x)
	vector<Mat> fImageBorderVectorA(3);
	vector<Mat>::iterator itAA = fImageBorderVectorA.begin();
	for (int i = 0; it != fImageBorderVector.end() && itAA != fImageBorderVectorA.end(); it++, itAA++, i++)
	{
		Mat roi(*it, Rect(hPatch / 2, vPatch / 2, darkChannel.cols, darkChannel.rows));

		(*itAA) = (*it) / A[i];

	}




	/*t(x)*/
	Mat darkChannelA(darkChannel.rows, darkChannel.cols, CV_32FC1);
	float omega = 0.9;//0<w<=1, 0.9
					  
	for (unsigned int r = 0; r < darkChannel.rows; r++)
	{
		for (unsigned int c = 0; c < darkChannel.cols; c++)
		{
			minPixel = 1.0;
			for (itAA = fImageBorderVectorA.begin(); itAA != fImageBorderVectorA.end(); itAA++)
			{
				Mat roi(*itAA, Rect(c, r, hPatch, vPatch));
				minMaxLoc(roi, &minTemp);
				minPixel = min(minPixel, minTemp);
			}


			darkChannelA.at<float>(r, c) = float(minPixel);
		}
	}
	Mat tx = 1.0 - omega*darkChannelA;

	//selective modifyt the transmission for the sky area

	switch (Y) {
	case(0): break;
	case(1): {

		Mat B1, G1, R1;
		vector<Mat> C1;
		split(fImage, C1);
		B1 = C1.at(0);
		G1 = C1.at(1);
		R1 = C1.at(2);
		double f, f0 = (A_mean - im_mean) / A_mean;
		for (int x = 0; x < image.rows; x++)
		{
			float  *data2 = tx.ptr<float>(x);
			const float *DB1 = B1.ptr<float>(x);
			const float *DG1 = G1.ptr<float>(x);
			const float *DR1 = R1.ptr<float>(x);
			for (int y = 0; y < image.cols; y++)
			{
				f = fabs((A[0] - DB1[y] + A[1] - DG1[y] + A[2] - DR1[y]) / 3);
				if (f < f0)
					data2[y] = data2[y] + (f0 - f) / f0;
				else
					data2[y] = data2[y];
			}
		}
	}
	}





	Size size(tx.cols*0.015, tx.rows*0.015);

	imshow("errosion", tx);
	//erode the transmission
	Mat element = getStructuringElement(MORPH_RECT, size);
	erode(tx, tx, element);
	imshow("errosion", tx);
	


	Mat tx1;
	//guided filter
	//guided image needed to be unified because tx is in(0, 1)
	guidedFilter(tx, fImage1, tx, 90, 0.001);
	imshow("guided", tx);

	/*J(x)*/
	float t0 = 0.1;//in He;s paper, t0 = 0.1
	Mat jx(image.rows, image.cols, CV_32FC3);
	for (size_t r = 0; r < jx.rows; r++)
	{
		for (size_t c = 0; c<jx.cols; c++)
		{
			jx.at<Vec3f>(r, c) = Vec3f((fImage.at<Vec3f>(r, c)[0] - A[0]) / max(tx.at<float>(r, c), t0) + A[0], (fImage.at<Vec3f>(r, c)[1] - A[1]) / max(tx.at<float>(r, c), t0) + A[1], (fImage.at<Vec3f>(r, c)[2] - A[2]) / max(tx.at<float>(r, c), t0) + A[2]);

		}
	}
	double ntime = ((double)getTickCount() - timestart) / cvGetTickFrequency();
	cout << "running time=" << ntime << "us" << endl;
	dst = Mat::zeros(jx.size(), jx.type());
	jx.convertTo(dst, -1, alpha, beta);
	namedWindow("dehazed", 1);
	imshow("dehazed", dst);
	Mat jx8U;
	dst.convertTo(jx8U, CV_8UC3, 255, 0);
	imwrite("dehazed.jpg", jx8U);
	waitKey(0);
	system("pause");
	return 0;
}