#include <array>

#include <sstream>
#include <complex>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

using namespace std;
using namespace Eigen;

int main(int argc, char * args[])
{
	if (argc < 3)
	{
		std::cout << "\nSyntax error - usage: " << args[0] << " <samplingFreq> <acquisitionTime>\n";
		return 0;
	}

	std::array<double, 8> freqs = {{1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000}};
	unsigned fs;
	unsigned N;
	unsigned acquisitionCycles;

	std::stringstream str;
	str << args[1] << " " << args[2];
	str >> std::scientific >> fs >> acquisitionCycles;

	double fundamentalPeriod = 1/freqs[0];
	double signalDuration = acquisitionCycles * fundamentalPeriod;

	//Discrete frequency axis
	N = fs * signalDuration;
	float freqGap = static_cast<float>(fs)/N;
	Eigen::VectorXf freq = Eigen::VectorXf::LinSpaced(N, 0, fs);

	std::cout << "\nSignal duration: " << std::scientific << signalDuration << " sec"
			<< "\nSampling freq: " << fs << " Hz"
			<< "\nN (num samples): " << N
			<< "\nFreq spacing: " << freqGap << "Hz\n";
	
	Eigen::FFT<float> fftCalc;

	//Time samples
	Eigen::VectorXf time = Eigen::VectorXf::LinSpaced(N, 0, signalDuration);
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
	fft.normalize();

	//Calculate FFT magnitude
	Eigen::VectorXf fftMagnitude = fft.array().abs();

	//PSD (magnitudeSquared)
	auto magnitudeSquared = fftMagnitude.array().pow(2);
	std::ofstream fftStream("fft.out", std::ios_base::out);
	if (fftStream.is_open())
	{
		fftStream	<< "timeSignal" << '\t' << '\t'
					<< "Frequency\t"	
					<< "FFT\t" 
					<< "fftMagnitude\t"
					<< "magnitudeSquared\n";

		for (int i=0; i < fft.size(); ++i)
		{
			fftStream << timeSignal(i) << "\t\t" 
				  << freq(i) << '\t'
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
	std::vector<std::pair<unsigned, float>> halfSpectrum;
	for (unsigned i=0; i < fftMagnitude.size()/2; ++i)
	{
		std::pair<unsigned, float> fftEntry;
		fftEntry.first = i;
		fftEntry.second = fftMagnitude(i);
		halfSpectrum.push_back(fftEntry);
	}

	//SORT by magnitude
	std::sort(
			std::begin(halfSpectrum), 
			std::end(halfSpectrum), 
			[] (const std::pair<unsigned, float>& fft1, const std::pair<unsigned, float> fft2) -> bool 
				{ return fft1.second > fft2.second; }
			);


	//User output
	std::cout << '\n';
	for (unsigned i=0; i<freqs.size(); ++i)
	{
		std::cout << "[" << halfSpectrum[i].first << "] " 
			<< "Freq: " << std::fixed << (halfSpectrum[i].first)*freqGap << " Hz"
			<< " Magnitude " << halfSpectrum[i].second << '\n';
	}

	return 0;
}
