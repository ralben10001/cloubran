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
	int classifier2;
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
int correct = 0;
int total = 0;

void do_predict(svmlines* classifier, FILE *output, struct model* model_, int startloc, int numlines, int numfeatures, int* prediction)
{

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
			i++;
			
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
			*prediction = predict_label;
		}

		if(predict_label == target_label)
			++correct;
		++total;
	}
	if(flag_predict_probability)
		free(prob_estimates);
}


void parse_command_line(int argc, char **argv);
void read_problem( svmlines* classifier, int numlines,int numfeatures, problem* prob, int pred);

struct feature_node *x_space;
struct parameter param;
int flag_cross_validation;
int nr_fold;
double bias;
int main(int argc, char *argv[])
{
	int* data;
	int * net;
	int datalen,netlen;
	struct problem prob, prob2;
	int numVMs = 1;
	if (argc > 1) {
		numVMs = atoi(argv[1]);
	}
	FILE* in;
	if (argc > 2) {
		in = fopen(argv[2],"r");
	} else {
		in = fopen("data1","r");
	}
	//FILE* in2 = fopen("net.in","r");
	FILE *output = fopen("test.out","w");
	char VMname[20];
	for (int ind = 0; ind < numVMs; ind++) {
		fscanf(in,"%s",VMname);
		fscanf(in,"%d",&datalen);
		//fscanf(in2,"%d",&netlen);
		fprintf(output,"%s ", VMname);
		data = new int[datalen];
		//net = new int[netlen];	
		for (int i = 0; i < datalen; i++) {
			fscanf(in,"%d",&data[i]);
		}
		/*for (int i = 0; i < netlen; i++) {
			fscanf(in2,"%d",&net[i]);
		}*/
		int n = 3;
		int numlines = datalen-n;
		int numruns = 1;
		prob.numpos = 0;
		svmlines* classifier = new svmlines[datalen-n];
		for (int i = 0; i < datalen-n; i++) {
			if (data[i+n]  > data[i+n-1]) {
				classifier[i].classifier = 1;
				if (i < numlines)
					prob.numpos++;
			}
			else
				classifier[i].classifier = -1;
			if (abs(data[i+n] -data[i+n-1]) > 5)
				classifier[i].classifier2 = 1;
			else
				classifier[i].classifier2 = -1;
			classifier[i].data = new int[2*n];
			classifier[i].data[0] = data[i];
			for (int j = 1; j < n; j++) {
				classifier[i].data[j] = data[i+j] - data[i+j-1];
			}
				classifier[i].data[n] = 0;
			for (int j = 1; j < n; j++) {
				classifier[i].data[n+j] = 0;//classifier[i].data[j]-classifier[i].data[j-1];
			}
		}
	
		prob.SV = new int*[101];
		prob.nSV = new int[101];
		for (int i = 0; i < 101; i++)
			prob.SV[i] = NULL;
	
		for (int s = 0; s < numruns; s++) {
			parse_command_line(argc, argv);
			read_problem(classifier,numlines+s-2,2*n,&prob,0);

			model_=train(&prob, &param);

			x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
			int prediction;
			do_predict(classifier, output, model_,numlines+s-2,1,2*n,&prediction);
			free(line);
			free(x);
			free(prob.y);
			free(prob.x);
			free(x_space);
			read_problem(classifier,numlines+s,2*n,&prob,prediction);
			free(line);
			//free(x);
			free(prob.y);
			free(prob.x);
			free(x_space);
			//free(line);
			if (classifier[numlines+s].classifier == 1)
				prob.numpos++;
			free_and_destroy_model(&model_);
		}
		printf("Accuracy = %g%% (%d/%d)\n",(double) correct/total*100,correct,total);
		fclose(output);
	}
	return 0;
}


void parse_command_line(int argc, char **argv)
{

	// default values
	param.solver_type = 2;
	param.C = 5;
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
void read_problem( svmlines* classifier, int numlines, int numfeatures, problem * prob, int pred)
{
	int max_index, i;
	long int elements, j;
	elements = numlines*(numfeatures+1);

	switch(pred) {
	case 0:
		prob->l = numlines;
		break;
	case 1:
		prob->l = prob->numpos;
		break;
	case -1:
		prob->l = numlines-prob->numpos;
		break;
	}
	prob->y = new double[prob->l];
	prob->x = Malloc(struct feature_node *,prob->l);
	x_space = Malloc(struct feature_node,elements+prob->l);
	
	prob->bias=bias;

	max_index = numfeatures; 
	int s = 0;
	for(i=0;i<numlines;i++)
	{ 
		if (pred == -classifier[i].classifier)
			continue;
		prob->x[s] = &x_space[s*(numfeatures+2)];
		feature_node* tmp = &x_space[s*(numfeatures+2)];
		prob->y[s] = classifier[i].classifier;

		for (j=0; j <numfeatures;j++) {
			tmp[j].index = j+1;
			tmp[j].value = classifier[i].data[j];
		}

		if(prob->bias >= 0)
			tmp[numfeatures].value = prob->bias;

		tmp[numfeatures+1].index = -1;
		s++;
	}

	if(prob->bias >= 0)
	{
		prob->n=max_index+1;
		for(i=1;i<prob->l;i++)
			(prob->x[i]-2)->index = prob->n; 
		j = (prob->n+1)*prob->l;
		x_space[j-2].index = prob->n;
	}
	else
		prob->n=max_index;

}

