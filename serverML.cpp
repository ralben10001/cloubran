#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "linear.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

struct feature_node *x;

typedef struct svmlines {
	int* data;
	int classifier;
} svmlines;
int max_nr_attr = 64;

struct model* model_;
int flag_predict_probability=0;

static char *line = NULL;
static int max_line_len;

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s type : set type of solver (default 1)\n"
	"	0 -- L2-regularized logistic regression\n"
	"	1 -- L2-regularized L2-loss support vector classification (dual)\n"	
	"	2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	4 -- multi-class support vector classification by Crammer and Singer\n"
	"	5 -- L1-regularized L2-loss support vector classification\n"
	"	6 -- L1-regularized logistic regression\n"
	"-c cost : set the parameter C (default 1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n" 
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" 
	"		where f is the primal function and pos/neg are # of\n" 
	"		positive/negative data (default 0.01)\n"
	"	-s 1, 3, and 4\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_inf <= eps*min(pos,neg)/l*|f'(w0)|_inf,\n"
	"		where f is the primal function (default 0.01)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}


static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void do_predict(svmlines* classifier, FILE *output, struct model* model_, int startloc, int numlines, int numfeatures)
{
	int correct = 0;
	int total = 0;

	int nr_class=get_nr_class(model_);
	double *prob_estimates=NULL;
	int j, n;
	int nr_feature=get_nr_feature(model_);
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	if(flag_predict_probability)
	{
		int *labels;

		if(model_->param.solver_type!=L2R_LR)
		{
			fprintf(stderr, "probability output is only supported for logistic regression\n");
			exit(1);
		}

		labels=(int *) malloc(nr_class*sizeof(int));
		get_labels(model_,labels);
		prob_estimates = (double *) malloc(nr_class*sizeof(double));
		fprintf(output,"labels");		
		for(j=0;j<nr_class;j++)
			fprintf(output," %d",labels[j]);
		fprintf(output,"\n");
		free(labels);
	}

	int linesread = startloc;
	while(linesread < startloc+numlines)
	{
		linesread++;
		int i = 0;
		int target_label, predict_label;
		int inst_max_index = 0; // strtol gives 0 if wrong format

		target_label = classifier[linesread].classifier;

		for (int j = 0; j < numfeatures; j++)
		{
			if(i>=max_nr_attr-2)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x,max_nr_attr*sizeof(struct feature_node));
			}

			x[i].index = j+1;
			
			x[i].value = classifier[linesread].data[j];
			
		}

		if(model_->bias>=0)
		{
			x[i].index = n;
			x[i].value = model_->bias;
			i++;
		}
		x[i].index = -1;

		if(flag_predict_probability)
		{
			int j;
			predict_label = (int)predict_probability(model_,x,prob_estimates);
			fprintf(output,"%d",predict_label);
			for(j=0;j<model_->nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else
		{
			predict_label = (int)predict(model_,x);
			fprintf(output,"%d %d\n",predict_label, target_label);
		}

		if(predict_label == target_label)
			++correct;
		++total;
	}
	printf("Accuracy = %g%% (%d/%d)\n",(double) correct/total*100,correct,total);
	if(flag_predict_probability)
		free(prob_estimates);
}


void parse_command_line(int argc, char **argv);
void read_problem( svmlines* classifier, int numlines,int numfeatures);
void do_cross_validation();

struct feature_node *x_space;
struct parameter param;
struct problem prob;
int flag_cross_validation;
int nr_fold;
double bias;
int main(int argc, char *argv[])
{
	int* data;
	int * net;
	int datalen,netlen;
	FILE* in = fopen("data.in","r");
	FILE* in2 = fopen("net.in","r");
	fscanf(in,"%d",&datalen);
	fscanf(in2,"%d",&netlen);
	data = new int[datalen];
	net = new int[netlen];
	for (int i = 0; i < datalen; i++) {
		fscanf(in,"%d",&data[i]);
	}
	for (int i = 0; i < netlen; i++) {
		fscanf(in,"%d",&net[i]);
	}
	int n = 3;
	svmlines* classifier = new svmlines[datalen-n];
	for (int i = 0; i < datalen-n; i++) {
		classifier[i].classifier = data[i+n]-data[i+n-1];
		classifier[i].data = new int[2*n];
		for (int j = 0; j < n; j++) {
			classifier[i].data[j] = data[i+j];
		}
		for (int j = 0; j < n; j++) {
			classifier[i].data[j+n] = net[i+j];
		}
	}
	
	FILE *output;
	int numlines = 5;
	int numruns = 5;
	output = fopen("test.out","w");
	if (argc > 1) {
		numlines = atoi(argv[3]);
		numruns = atoi(argv[4]);
	}
	
	for (int s = 0; s < numruns; s++) {
		parse_command_line(argc, argv);
		read_problem(classifier,numlines+s,2*n);
		const char *error_msg;

		error_msg = check_parameter(&prob,&param);

		if(error_msg)
		{
			fprintf(stderr,"Error: %s\n",error_msg);
			exit(1);
		}

		if(flag_cross_validation)
		{
			do_cross_validation();
		}
		else
		{
			model_=train(&prob, &param);
		}

		x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
		do_predict(classifier, output, model_,numlines+s,1,2*n);
		free_and_destroy_model(&model_);
		free(line);
		free(x);
		free(prob.y);
		free(prob.x);
		free(x_space);
		//free(line);
	}
	fclose(output);
	return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double *target = new double(prob.l);
	FILE* out = fopen("out.txt","w");
	cross_validation(&prob,&param,nr_fold,target);

	for(i=0;i<prob.l;i++) {
		fprintf(out, "%d %d\n", target[i], prob.y[i]);
		printf("%d %d\n",target[i],prob.y[i]);
		if(target[i] == prob.y[i])
			++total_correct;
	}
	printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	fclose(out);
	free(target);
}

void parse_command_line(int argc, char **argv)
{

	// default values
	param.solver_type = 2;
	param.C = 1;
	param.eps = INF; // see setting below
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	flag_cross_validation = 0;
	bias = 10;

	// parse options

	// determine filenames

	if(param.eps == INF)
	{
		if(param.solver_type == L2R_LR || param.solver_type == L2R_L2LOSS_SVC)
			param.eps = 0.01;
		else if(param.solver_type == L2R_L2LOSS_SVC_DUAL || param.solver_type == L2R_L1LOSS_SVC_DUAL || param.solver_type == MCSVM_CS)
			param.eps = 0.1;
		else if(param.solver_type == L1R_L2LOSS_SVC || param.solver_type == L1R_LR)
			param.eps = 0.01;
	}
}

// read in a problem (in libsvm format)
void read_problem( svmlines* classifier, int numlines, int numfeatures)
{
	int max_index, i;
	long int elements, j;


	prob.l = numlines;
	elements = numlines*numfeatures;
	
	prob.bias=bias;
	prob.y = new double[prob.l];
	prob.x = Malloc(struct feature_node *,prob.l);
	x_space = Malloc(struct feature_node,elements+prob.l);

	max_index = numfeatures+1; 
	for(i=0;i<prob.l;i++)
	{ 
		prob.x[i] = &x_space[i*(numfeatures+1)];
		prob.y[i] = classifier[i].classifier;

		for (j=0; j <numfeatures;j++) {
			x_space[j].index = j+1;
			x_space[j].value = classifier[i].data[j];
		}

		if(prob.bias >= 0)
			x_space[numfeatures+1].value = prob.bias;

		x_space[numfeatures+1].index = -1;
	}

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n; 
		x_space[j-2].index = prob.n;
	}
	else
		prob.n=max_index;

}
