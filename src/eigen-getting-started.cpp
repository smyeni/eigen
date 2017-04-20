#include <array>
#include <algorithm>
#include <complex>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

using namespace std;
using namespace Eigen;

int main()
{
	std::array<unsigned, 8> freqs = {{1, 3, 5, 7, 9, 11, 13, 15}};
	unsigned duration = 1; //7sec DAQ duration
	unsigned fs = 256;
	const unsigned N = fs/freqs[0]; //samples collected over one lowest freq cycle
	deltaFreq = fs/(N-1);
	
	Eigen::FFT<float> fftCalc;

	//Time samples
	Eigen::VectorXf time = Eigen::VectorXf::LinSpaced(N, 0, duration);
	std::cout << "\nNumber of time points: " << time.size() 
			  << " [" << time(0) << ".." << time(time.size()-1) << "]\n";

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
	Eigen::VectorXcf fft(N);
	fftCalc.fwd(fft, timeSignal);
	
	//Normalize FFT
	//fft.normalize();

	//Calculate FFT magnitude
	Eigen::VectorXf fftMagnitude = fft.array().abs();

	//PSD (magnitudeSquared)
	auto magnitudeSquared = fftMagnitude.array().pow(2);
	std::ofstream fftStream("fft.out", std::ios_base::out);
	if (fftStream.is_open())
	{
		fftStream << "timeSignal" << '\t' << '\t' 
			  << "FFT" << '\t' << '\t' 
			  << "fftMagnitude" << '\t' << '\t' 
			  << "magnitudeSquared\n";

		for (int i=0; i < fft.size(); ++i)
		{
			fftStream << timeSignal(i) << "\t\t" 
				  << fft(i).real() << (fft(i).imag() < 0 ? '-' : '+') << std::abs(fft(i).imag()) << "i\t"
				  << fftMagnitude(i) << '\t' 
				  << magnitudeSquared(i) << '\n';
		}

		fftStream.close();
	}
	else
	{
		std::cout << "\nFailed to open output stream, no new FFT file\n";
	}

	//Half spectrum
	std::vector<std::pair<unsigned, float>> fftPairs;
	for (unsigned i=0; i < fftMagnitude.size()/2; ++i)
	{
		std::pair<unsigned, float> entry;
		entry.first = i;
		entry.second = fftMagnitude(i);
		fftPairs.push_back(entry);
	}

	//SORT by magnitude
	std::sort(
			std::begin(fftPairs), 
			std::end(fftPairs), 
			[] (const std::pair<unsigned, float>& a, const std::pair<unsigned, float> b) -> bool 
				{ return a.second > b.second; }
			);


	//User output
	std::cout << '\n';
	for (unsigned i=0; i<freqs.size(); ++i)
	{
		std::cout << "[" << fftPairs[i].first << "] " << fftPairs[i].second << '\n';
	}

	return 0;
}
