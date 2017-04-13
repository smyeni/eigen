#include <complex>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

using namespace std;
using namespace Eigen;

int main()
{
	unsigned freqs[] = {1, 3, 5, 7, 9, 11, 13, 15};
	unsigned duration = 1; //7sec DAQ duration
	unsigned fs = 256;
	const unsigned N = fs/freqs[0]; //samples collected over one lowest freq cycle

	Eigen::FFT<float> fft;

	//Time samples
	Eigen::VectorXf time = Eigen::VectorXf::LinSpaced(N, 0, duration);
	std::cout << "\nNumber of time points: " << time.size() 
			  << "[" << time(0) << ".." << time(time.size()-1) << "]\n";

	Eigen::VectorXf timeSignal(N);
	for (unsigned i=0; i<N; ++i)
	{
		timeSignal(i) = sin(2*M_PI*freqs[0]*time(i)) + 
						sin(2*M_PI*freqs[1]*time(i))/3 +
						sin(2*M_PI*freqs[2]*time(i))/5 +
						sin(2*M_PI*freqs[3]*time(i))/7 +
						sin(2*M_PI*freqs[4]*time(i))/9 + 
						sin(2*M_PI*freqs[5]*time(i))/11 + 
						sin(2*M_PI*freqs[6]*time(i))/13 + 
						sin(2*M_PI*freqs[7]*time(i))/15;
	}

	//Calculate FFT
	Eigen::VectorXcf freqDomain(N);
	fft.fwd(freqDomain, timeSignal);
	
	//Normalize FFT
	freqDomain.normalize();

	//PSD
	Eigen::VectorXf magnitude(freqDomain.size());
   	for (int k=0; k < freqDomain.size(); ++k)
	{
		magnitude(k) = std::abs(freqDomain(k));
	}

	Eigen::VectorXf magnitudeVec = freqDomain.array().abs();

	//PSD
	auto psd = magnitudeVec.array().pow(2);
	std::ofstream fftStream("fft.out", std::ios_base::out);
	if (fftStream.is_open())
	{
		fftStream << "timeSignal" << '\t' << '\t' 
				  << "freqDomain" << '\t' << '\t' 
				  << "magnitude" << '\t' << '\t' 
				  << "magnitude2" << '\t' << '\t' 
				  << "psd" << '\n';

		for (int i=0; i < freqDomain.size(); ++i)
		{
			fftStream << timeSignal(i) << '\t' << '\t' 
					  << freqDomain(i) << '\t' << '\t' 
					  << magnitude(i) << " : " 
					  << magnitudeVec(i) << '\t' << '\t' 
					  << psd(i) << '\n';
		}

		fftStream.close();
	}
	else
	{
		std::cout << "\nFailed to open output stream, no new FFT file\n";
	}

	return 0;
}
