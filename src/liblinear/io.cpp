
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <ctype.h>
#include <errno.h>

#include "io.h"
#include "linear.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))


static char *line = NULL;
static int max_line_len;

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var)do\
{\
	if (fscanf(_stream, _format, _var) != 1)\
	{\
		fprintf(stderr, "ERROR: fscanf failed to read the model\n");\
		EXIT_LOAD_MODEL()\
	}\
}while(0)
// EXIT_LOAD_MODEL should NOT end with a semicolon.
#define EXIT_LOAD_MODEL()\
{\
	setlocale(LC_ALL, old_locale);\
	free(model_->label);\
	free(model_);\
	free(old_locale);\
	return NULL;\
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				EXIT_LOAD_MODEL()
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			FSCANF(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			FSCANF(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			FSCANF(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				FSCANF(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			EXIT_LOAD_MODEL()
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			FSCANF(fp, "%lf ", &model_->w[i*nr_w+j]);
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
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


void exit_input_error(int line_num)
{
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    exit(1);
}


// read in a problem (in libsvm format)
struct problem *read_problem(const char *filename, double bias)
{
    struct problem *prob_ = Malloc(problem,1);
    struct problem prob = *prob_;

    int max_index, inst_max_index, i;
    size_t elements, j;
    FILE *fp = fopen(filename,"r");
    char *endptr;
    char *idx, *val, *label;

    if(fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        exit(1);
    }

    prob.l = 0;
    elements = 0;
    line = Malloc(char,max_line_len);
    while(readline(fp)!=NULL)
    {
        char *p = strtok(line," \t"); // label

        // features
        while(1)
        {
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            elements++;
        }
        elements++; // for bias term
        prob.l++;
    }
    rewind(fp);

    prob.bias=bias;

    prob.y = Malloc(double,prob.l);
    prob.x = Malloc(struct feature_node *,prob.l);
    struct feature_node *x_space = Malloc(struct feature_node,elements+prob.l);

    max_index = 0;
    j=0;
    for(i=0;i<prob.l;i++)
    {
        inst_max_index = 0; // strtol gives 0 if wrong format
        readline(fp);
        prob.x[i] = &x_space[j];
        label = strtok(line," \t\n");
        if(label == NULL) // empty line
            exit_input_error(i+1);

        prob.y[i] = strtod(label,&endptr);
        if(endptr == label || *endptr != '\0')
            exit_input_error(i+1);

        while(1)
        {
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;

            errno = 0;
            x_space[j].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                exit_input_error(i+1);
            else
                inst_max_index = x_space[j].index;

            errno = 0;
            x_space[j].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);

            ++j;
        }

        if(inst_max_index > max_index)
            max_index = inst_max_index;

        if(prob.bias >= 0)
            x_space[j++].value = prob.bias;

        x_space[j++].index = -1;
    }

    if(prob.bias >= 0)
    {
        prob.n=max_index+1;
        for(i=1;i<prob.l;i++)
            (prob.x[i]-2)->index = prob.n;
        x_space[j-2].index = prob.n;
    }
    else
    {
        prob.n = max_index;
    }

    fclose(fp);
    return prob_;
}
