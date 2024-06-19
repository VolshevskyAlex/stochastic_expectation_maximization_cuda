#define NOMINMAX
#include <glew.h>
#include <wglew.h>
#include <freeglut.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_gl_interop.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>
#include <memory>

#include "paramgl2.h"

#include "em.h"
#include "simulation.h"
#include "gen_random_clouds.h"
#include "render_tables.h"

float camera_trans[] = {0, 0, -3.5};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -3.5};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1f;
const int timer_interval = 10;

int ox, oy;
int buttonState = 0;

bool displaySliders = true;
std::unique_ptr<ParamListGL> params;
distr_view d_view_et;

// параметры модели
int    data_len = 1000000;
float  density;				// доля видимых точек
int    cluster_count;
float  max_sigma;
int    random_seed;

bool step_mode;
int wait4continue;

int main_wnd;

GLuint VBO;
GLuint tex0, tex1;
GLuint prog;
GLuint threshold_p;
cudaGraphicsResource *cuda_VBO;
float *dv_rand;

std::unique_ptr<Simulation> sim;

#define STRINGIFY(A) #A

const char *vertex_shader = STRINGIFY(
	varying float rnd_v;
	void main(void)
	{
		vec4 v = gl_Vertex;
		rnd_v = v.w;
		v.w = 1.0;
		gl_Position = gl_ModelViewProjectionMatrix * v;
	});

const char *fragment_shader = STRINGIFY(
	uniform float threshold;
	varying float rnd_v;
	void main(void)
	{
		if (rnd_v > threshold)
			discard;
		gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
	});

bool update_camera();
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void display();
void render_points();
void render_ellipses();
void sim_reset();

bool update_camera()
{
	const float eps = 1e-6f;
	bool upd = false;
	for (int c = 0; c < 3; ++c)
	{
		if (fabs(camera_trans_lag[c] - camera_trans[c]) > eps ||
			fabs(camera_rot_lag[c] - camera_rot[c]) > eps)
		{
			camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
			camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
			upd = true;
		}
	}
	return upd;
}

void mouse(int button, int state, int x, int y)
{
	if (displaySliders)
	{
		if (params->Mouse(x, y, button, state))
		{
			glutPostRedisplay();
			return;
		}
	}
	if (state == GLUT_DOWN)
	{
		buttonState = 1;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}
	ox = x;
	oy = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
    if (displaySliders)
    {
        if (params->Motion(x, y))
        {
            ox = x;
            oy = y;
            glutPostRedisplay();
            return;
        }
    }

	if (buttonState)
	{
		int	mods = glutGetModifiers();
		float dx, dy;
		dx = (float)(x - ox);
		dy = (float)(y - oy);

		if (mods & GLUT_ACTIVE_CTRL)
		{
			// zoom
			camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
		}
		else if (mods & GLUT_ACTIVE_SHIFT)
		{
			// translate
			camera_trans[0] += dx / 100.0f;
			camera_trans[1] -= dy / 100.0f;
		}
		else
		{
			// rotate
			camera_rot[0] += dy / 5.0f;
			camera_rot[1] += dx / 5.0f;
		}
		ox = x;
		oy = y;
		if (update_camera())
		{
			glutPostRedisplay();
		}
	}
}

void display()
{
    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    // cube
    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(2.0);

	render_points();
	render_ellipses();

    if (displaySliders)
    {
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
        params->Render(0, 0);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }

    glutSwapBuffers();
    glutReportErrors();
}

void log(GLhandleARB obj)
{
	GLint len = 0;
	glGetObjectParameterivARB(obj, GL_OBJECT_INFO_LOG_LENGTH_ARB, &len);
	if (len <= 1)
		return;

	std::string bf;
	bf.resize(len);
	GLsizei sz;
	glGetInfoLogARB(obj, len, &sz, &bf[0]);
	OutputDebugString(bf.c_str());
}

GLuint create_shader(GLenum shaderType, const char *src_p)
{
	GLuint shader  = glCreateShaderObjectARB(shaderType);
	glShaderSourceARB(shader, 1, &src_p, NULL);
	glCompileShaderARB(shader);
	GLint status;
	glGetObjectParameterivARB(shader, GL_OBJECT_COMPILE_STATUS_ARB, &status);
	log(shader);
	int err = glGetError();

	return shader;
}

void disable_vert_sync()
{
    if (wglewIsSupported("WGL_EXT_swap_control"))
    {
        wglSwapIntervalEXT(0);
    }
}

void key_fn(unsigned char k, int, int)
{
	switch (k)
	{
	case 'r':
		sim_reset();
		step_mode = false;
		sim->start();
		break;
	case 's':
		sim_reset();
		step_mode = true;
		sim->start();
		break;
	case ' ':
		if (sim->is_run())
			sim->sim_continue();
		break;
	case 27:
		exit(EXIT_SUCCESS);
	}
}

struct distr_view_wnd
{
	int wnd_;
	std::string name_;
	void (*disp_fn_)();
	bool auto_reshape_;
	bool visible_;

	distr_view_wnd(const char *name, void (*disp_fn)()) :
		name_(name), disp_fn_(disp_fn), wnd_(), visible_(false), auto_reshape_(true)
	{
	}

	void adjust4data(distr_view &view)
	{
		int tb_width, tb_height;
		view.GetMaxSize(tb_width, tb_height);
		int w = glutGet(GLUT_SCREEN_WIDTH) - 20;
		int sep = view.sep_size_;
		int col_cnt = std::min((int)view.t_.size(), w / (tb_width + sep));
		int width  = (tb_width +sep) * col_cnt;
		int height = (tb_height+sep) * ((int)(view.t_.size()+col_cnt-1) / col_cnt);
		int cur = glutGetWindow();
		if (!wnd_)
		{
			glutInitWindowPosition(0, 725);
			glutInitWindowSize(width, height);
			wnd_ = glutCreateWindow(name_.c_str());
			disable_vert_sync();
			glutDisplayFunc(disp_fn_);
			glutReshapeFunc(reshape);
			glutKeyboardFunc(key_fn);
		}
		else if (auto_reshape_)
		{
			glutSetWindow(wnd_);
			glutReshapeWindow((tb_width+sep) * col_cnt, (tb_height+sep) * ((view.t_.size()+col_cnt-1) / col_cnt) );
		}
		glutSetWindow(cur);
	}
	void update_wnd()
	{
		int cur = glutGetWindow();
		glutSetWindow(wnd_);
		glutPostRedisplay();
		glutSetWindow(cur);
	}
	void show(bool s)
	{
		if (wnd_)
		{
			int cur = glutGetWindow();
			glutSetWindow(wnd_);
			if (s)
				glutShowWindow();
			else
				glutHideWindow();

			glutSetWindow(cur);
		}
	}
	static void reshape(int w, int h)
	{
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glViewport(0, 0, w, h);
	}
};

void display_et()
{
	glDisable(GL_DEPTH_TEST);
	d_view_et.Render();
	glutSwapBuffers();
}

void display_d()
{
	glDisable(GL_DEPTH_TEST);
	distr_view &view = sim->lock_distr_view();
	view.Render();
	sim->unlock_distr_view();
	glutSwapBuffers();
}

distr_view_wnd distr_view_et_wnd("distribution parameters", display_et);		// окно эталонных таблиц
distr_view_wnd distr_view_wnd("parameters estimation", display_d);				// окно таблиц оценки параметров

void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowPosition(0, 0);
    glutInitWindowSize(1100, 700);
    main_wnd = glutCreateWindow("CUDA Expecation Maximization");

    glewInit();
	disable_vert_sync();

    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object"))
    {
    }

	glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);

	GLuint vert_sh = create_shader(GL_VERTEX_SHADER_ARB  , vertex_shader);
	GLuint frag_sh = create_shader(GL_FRAGMENT_SHADER_ARB, fragment_shader);
	prog = glCreateProgramObjectARB();
	glAttachObjectARB(prog, vert_sh);
	glAttachObjectARB(prog, frag_sh);
	glLinkProgramARB(prog);
	GLenum err = glGetError();
	threshold_p = glGetUniformLocation(prog, "threshold");

	GLint status;
	glGetObjectParameterivARB(prog, GL_OBJECT_LINK_STATUS_ARB, &status);
	log(prog);
	err = glGetError();

	glGenBuffersARB(1, &VBO);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, VBO);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, data_len*4*sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

    glutReportErrors();
}

void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void render_points()
{
	glUseProgramObjectARB(prog);
	glUniform1f(threshold_p, density);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_POINTS, 0, data_len);
	glDisableClientState(GL_VERTEX_ARRAY);
	glUseProgramObjectARB(0);
}

void render_ellipses()
{
	vec_shapes &a = sim->lock_shape_list();
	for (vec_shapes::iterator it = a.begin(), e = a.end(); it!=e; ++it)
	{
		(*it)->Render();
	}
	sim->unlock_shape_list();
}

template <class T>
class Param_nf : public Param<T>
{
	void (*notify_)();
public:
	Param_nf(const char *name, T value, T min_v, T max_v, T step, T *ptr, void (*fn)()) :
		Param<T>(name, value, min_v, max_v, step, ptr),
		notify_(fn)
	{
	}
	void SetPercentage(float p)
	{
		T prev = GetValue();
		Param<T>::SetPercentage(p);
		if (GetValue()!=prev)
		{
			notify_();
		}
	}
	void Reset()
	{
		T prev = GetValue();
		Param<T>::Reset();
		if (GetValue()!=prev)
		{
			notify_();
		}
	}
};

void lanch_cvt(float4 *dst, float *src1, int src1_pitch, float *src2, int data_len);

void update_VBO()
{
	if (!dv_rand)
	{
	cudaError_t err = cudaMalloc((void**)&dv_rand, data_len*sizeof(float));

	curandGenerator_t prngGPU;
	curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(prngGPU, 1234);
	curandGenerateUniform(prngGPU, dv_rand, data_len);
	curandDestroyGenerator(prngGPU);
	}

	float4 *d;
	size_t len;
	cudaGraphicsMapResources(1, &cuda_VBO, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d, &len, cuda_VBO);
	lanch_cvt(d, sim->dv_data_, sim->data_pitch_, dv_rand, sim->data_len_);
	cudaGraphicsUnmapResources(1, &cuda_VBO, 0);
}

void sim_reset()
{
	sim->stop();

	vec_shapes &a = sim->lock_shape_list();
	a.clear();
	sim->unlock_shape_list();

	distr_view &view = sim->lock_distr_view();
	view.t_.clear();
	sim->unlock_distr_view();

	distr_view_wnd.show(false);
	glutSetWindow(main_wnd);
}

void update_distr_params()
{
	sim_reset();

	std::vector<float> &colors = sim->distr_colors_;
	colors.resize(3*cluster_count);
	std::vector<float> colors_w(3*cluster_count);
	srand(random_seed);
	for (int i = 0; i < cluster_count; i++)
	{
		double c[3];
		double max_c = 0;
		for (int j = 0; j < 3; j++)
		{
			c[j] = (double)rand() / (RAND_MAX+1);
			max_c = std::max(max_c, c[j]);
		}
		for (int j = 0; j < 3; j++)
		{
			colors[i*3+j] = (float)(c[j] / max_c);
			colors_w[i*3+j] = 1.f;
		}
	}

	sim->cluster_cnt_ = cluster_count;
	std::vector<distr_gauss_params<3> > gmm;
	fill_data4simulation(sim->dv_data_, sim->data_pitch_, sim->data_len_, sim->cluster_cnt_, random_seed, 0.3*max_sigma, max_sigma, gmm);
	update_VBO();

	d_view_et.Update(gmm, colors_w.data());
	distr_view_et_wnd.adjust4data(d_view_et);
	distr_view_et_wnd.update_wnd();
}

void initSliders()
{
	params.reset(new ParamListGL("misc"));
	params->AddParam(new Param<float>   ("density"      ,  0.05f, 0.0f, 1.0f, 0.01f, &density));
	params->AddParam(new Param_nf<int>  ("cluster count",     8, 2,  18, 1, &cluster_count, update_distr_params));
	params->AddParam(new Param_nf<float>("sigma"        , 0.05f, 0.02f,  0.2f, 0.005f, &max_sigma, update_distr_params));
	params->AddParam(new Param_nf<int>  ("random seed"  ,     50, 0, 100, 1, &random_seed  , update_distr_params));
}

void timer_fn(int value)
{
	bool cam_upd = update_camera();
	bool sim_upd = !wait4continue && sim->is_iteration_completed();

	if (cam_upd || sim_upd)
		glutPostRedisplay();

	if (sim_upd)
	{
		distr_view &view = sim->lock_distr_view();
		if (! view.t_.empty())
		{
			if (!distr_view_wnd.visible_)
			{
				distr_view &view = sim->lock_distr_view();
				distr_view_wnd.adjust4data(view);
				sim->unlock_distr_view();

				distr_view_wnd.show(true);
			}
			else
				distr_view_wnd.update_wnd();
		}
		sim->unlock_distr_view();

		wait4continue = sim->is_att_completed() ? 5 : 1;
	}
	else if (!(step_mode && sim->is_att_completed()) && wait4continue && !--wait4continue)
	{
		sim->sim_continue();
	}
	glutTimerFunc(timer_interval, timer_fn, 0);
}

void mainMenu(int i)
{
	key_fn((unsigned char)i, 0, 0);
}

void initMenus()
{
	glutCreateMenu(mainMenu);
	glutAddMenuEntry("Run     [r]", 'r');
	glutAddMenuEntry("By step [s]", 's');
	glutAddMenuEntry("Next iteration (space)", ' ');
	glutAddMenuEntry("Quit (esc)", 27);
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void cleanup()
{
	cudaGraphicsUnregisterResource(cuda_VBO);
	cudaFree(dv_rand);
}

int main(int argc, char **argv)
{
	cudaDeviceProp cuda_props;
	if (cudaGetDeviceProperties(&cuda_props, 0)!=cudaSuccess)
	{
		printf("cudaGetDeviceProperties FAIL\n");
		return 0;
	}
	printf("\"%s\" compatibility %d.%d\n", cuda_props.name, cuda_props.major, cuda_props.minor);
	if (cuda_props.major < 2)
	{
		printf("compatibility >=2.0 required\n");
		return 0;
	}

	initGL(&argc, argv);
	if (cudaGLSetGLDevice(0)!=cudaSuccess)
	{
		printf("cudaGLSetGLDevice FAIL\n");
		return 0;
	}

	cudaGraphicsGLRegisterBuffer(&cuda_VBO, VBO, cudaGraphicsMapFlagsWriteDiscard);

	sim.reset(new Simulation(data_len));

	initMenus();
	initSliders();
	update_distr_params();
	update_VBO();

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutTimerFunc(timer_interval, timer_fn,0);
	glutKeyboardFunc(key_fn);

	atexit(cleanup);
	glutMainLoop();

	sim.reset();
	params.reset();
}
