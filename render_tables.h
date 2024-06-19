#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <memory>

#include "em.h"

struct gl_text_params
{
	void *font_;
	int font_h_;
	float color_[3];
	float bg_color_[3];
};

class gl_table
{
	int width_;
	int height_;
	gl_text_params params_;
	std::vector<std::string> rows_;

public:
	gl_table(std::istream &, const gl_text_params &);
	void draw_at(float x, float y, float w, float h);

	friend class distr_view;
};

class distr_view
{
public:
	std::vector<std::unique_ptr<gl_table> > t_;
	float bg_color_[3];
	int sep_size_;

	distr_view();
	void Update(const std::vector<distr_gauss_params<3> > &g, const float *colors);
	void Render();
	void GetMaxSize(int &width, int &height);
};
