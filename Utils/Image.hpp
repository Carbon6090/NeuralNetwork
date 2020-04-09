#pragma once

#include <vector>
#include <fstream>
#include <string>

class Image {
	// структура для пикселя
	struct Pixel {
		unsigned char r;
		unsigned char g;
		unsigned char b;
	};

	int width; // ширина картинки
	int height; // высота картинки
	std::vector<Pixel> data; // вектор пикселей

public:
	Image(int width, int height);

	void SetPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b); // изменение пикселя
	void Save(const std::string& name); // сохранение
};

Image::Image(int width, int height) {
	this->width = width;
	this->height = height;
	data.resize(width * height);
}

// изменение пикселя
void Image::SetPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b) {
	data[(height - 1 - y) * width + x] = { b, g, r };
}

// сохранение
void Image::Save(const std::string& name) {
	int paddedsize = (width*height) * sizeof(Pixel);

	unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
	unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};

	bmpfileheader[ 2] = (unsigned char)(paddedsize    );
	bmpfileheader[ 3] = (unsigned char)(paddedsize>> 8);
	bmpfileheader[ 4] = (unsigned char)(paddedsize>>16);
	bmpfileheader[ 5] = (unsigned char)(paddedsize>>24);

	bmpinfoheader[ 4] = (unsigned char)(width    );
	bmpinfoheader[ 5] = (unsigned char)(width>> 8);
	bmpinfoheader[ 6] = (unsigned char)(width>>16);
	bmpinfoheader[ 7] = (unsigned char)(width>>24);
	bmpinfoheader[ 8] = (unsigned char)(height    );
	bmpinfoheader[ 9] = (unsigned char)(height>> 8);
	bmpinfoheader[10] = (unsigned char)(height>>16);
	bmpinfoheader[11] = (unsigned char)(height>>24);

	std::ofstream out(name.c_str(), std::ios::out | std::ios::binary);
	out.write((const char*)bmpfileheader, 14);
	out.write((const char*)bmpinfoheader, 40);
	out.write((const char*)data.data(), paddedsize);
	out.flush();
	out.close();
}