#ifndef LIBLINEAR_IO_H_
#define LIBLINEAR_IO_H_

#ifdef __cplusplus
extern "C" {
#endif

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

struct problem *read_problem(const char *filename, double bias);

#ifdef __cplusplus
}
#endif

#endif //LIBLINEAR_IO_H_
