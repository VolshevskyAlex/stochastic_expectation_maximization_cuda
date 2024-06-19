#define NOMINMAX

#include <sstream>
#include <iomanip>

#include "render_tables.h"

#include <freeglut.h>

gl_table::gl_table(std::istream &is, const gl_text_params &p) :
	params_(p)
{
	width_  = 0;
	height_ = 0;
	for (;;)
	{
		std::string s;
		std::getline(is, s);
		if (s.empty())
			break;
		rows_.push_back(s);
		width_   = std::max(width_, glutBitmapLength(params_.font_, (const unsigned char*)s.c_str()));
		height_ += params_.font_h_;
	}
}

void gl_table::draw_at(float x, float y, float w, float h)
{
	glColor3fv(params_.bg_color_);
	glRectf(x, y, x+w, y+h);
	glColor3fv(params_.color_);
	for (size_t i = 0; i < rows_.size(); i++)
	{
		y += params_.font_h_;
		glRasterPos2f(x, y);
		const char *s = rows_[i].c_str();
		while (*s)
		{
	        glutBitmapCharacter(params_.font_, *s++);
		}
	}
}

distr_view::distr_view() :
	sep_size_(5)
{
	bg_color_[0] = 0.f;
	bg_color_[1] = 0.f;
	bg_color_[2] = 0.f;
}

void distr_view::Render()
{
	int tb_width;
	int tb_height;
	GetMaxSize(tb_width, tb_height);

	int win_width  = glutGet(GLUT_WINDOW_WIDTH);
	int win_height = glutGet(GLUT_WINDOW_HEIGHT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, win_width, 0, win_height, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, (GLfloat)(win_height - 1.0), 0.0);
	glScalef(1.0, -1.0, 1.0);
	int col_count = std::max(win_width / (tb_width+sep_size_), 1);

	glClearColor(bg_color_[0], bg_color_[1], bg_color_[2], 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	for (size_t i = 0; i < t_.size(); i++)
	{
		int nx = (i % col_count) * (tb_width +sep_size_);
		int ny = (i / col_count) * (tb_height+sep_size_);
		t_[i]->draw_at((float)nx, (float)ny, (float)tb_width, (float)tb_height);
	}
}

void distr_view::Update(const std::vector<distr_gauss_params<3> > &g, const float *colors)
{
	t_.clear();
	gl_text_params text_p;
	text_p.color_[0] = 0.f;
	text_p.color_[1] = 0.f;
	text_p.color_[2] = 0.f;
	text_p.font_ = GLUT_BITMAP_9_BY_15;
	text_p.font_h_ = 15;

	for (size_t i = 0; i < g.size(); i++)
	{
		text_p.bg_color_[0] = colors[3*i+0];
		text_p.bg_color_[1] = colors[3*i+1];
		text_p.bg_color_[2] = colors[3*i+2];
		enum {N=3};
		const int num_width = 10;
		const distr_gauss_params<N> &it = g[i];
		std::stringstream os;
		os << std::fixed;
		os << "tau=" << std::setw(num_width) << it.tau << "\n";

		for (int i = 0; i < N; i++)
		{
			os << "mu[=" << i << "]=" << std::setw(num_width) << it.mu[i] << "\n";
		}
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				os << std::setw(num_width) << it.cov[i*N+j];
			}
			os << "\n";
		}
		os.seekg(0);
		t_.push_back(std::unique_ptr<gl_table>(new gl_table(os, text_p)));
	}
}

void distr_view::GetMaxSize(int &width, int &height)
{
	width  = 0;
	height = 0;
	for (size_t i = 0; i < t_.size(); i++)
	{
		gl_table *it = t_[i].get();
		width  = std::max(width , it->width_);
		height = std::max(height, it->height_);
	}
}
