#include <stdio.h>
#include <stdlib.h>
#include <argp.h>
#include <stdbool.h>
#include <string.h>


static unsigned int N; /* Number of particles */
#define NSTATES (6)    /* Number of states */

#define S  0  // Susceptible
#define E  1  // Exposed (will become infected)
#define I  2  // Infectious
#define Q  3  // Quarantined (Infected but not infectious)
#define RI 4  // Recovered after being infectious
#define RQ 5  // Recovered after being quarantined
#define RANDOM_MAX (2147483647)

/***
 * Velocity Profile Struct: stores velocities and the points in time
 * after which each velocity applies
 ***/
struct velocity_profile {
  int size;
  double *times;
  double *velocities;
};

/***
 * Quarantine Ratio (rQ) Struct: stores the ratio of infectious
 * quarantined (rQ) and the points in time after which each rQ applies
 ***/
struct rQ_profile {
  int size;
  double *times;
  double *quarantine_ratios;
};

/***
 * Probability profile Struct: stores the infect probability for each
 * portion of population and the points in time after which each rQ
 * applies
 ***/
struct probability_profile {
  int n_times;
  int *n_probs;
  double *times;
  double **population;
  double **probability;
};


/***
 * Open file, catch error if open fails
 ***/
FILE *
uopen(char fname[], char *flags)
{
  FILE *fp;
  fp = fopen(fname, flags);
  if(fp == NULL) {
    fprintf(stderr, " Error: fopen() on file \"%s\" with flags \"%s\" returned NULL\n", fname, flags);
    exit(2);
  }
  return fp;
}

/***
 * Allocate a buffer, catch error if allocation fails
 ***/
void *
ualloc(size_t size)
{
  void *ptr = malloc(size);
  if(ptr == NULL) {
    fprintf(stderr, " Error: malloc() for \"%lu\" bytes returned NULL\n", size);
    exit(3);
  }
  return ptr;
}

/***
 * Reads a file and returns a velocity profile structure. Converts
 * each line to two floats (time and velocity) stopping after the
 * first failure to convert.
 ***/
struct velocity_profile *
load_velocity_profile(char fname[])
{
  FILE *fp = uopen(fname, "r");
  int nlines = 0;
  while(!feof(fp)) {    
    char line[1024];
    fgets(line, 1024, fp);
    nlines++;
    double x, y;
    int n = sscanf(line, "%lf %lf", &x, &y);
    if(n != 2)
      break;
  }
  nlines--;
  rewind(fp);
  struct velocity_profile *vp = ualloc(sizeof(struct velocity_profile));
  vp->size = nlines;
  vp->times = ualloc(nlines*sizeof(double));
  vp->velocities = ualloc(nlines*sizeof(double));
  for(int i=0; i<nlines; i++) {
    char line[1024];
    fgets(line, 1024, fp);
    sscanf(line, "%lf %lf", &vp->times[i], &vp->velocities[i]);
  }
  fclose(fp);
  return vp;
}

/***
 * Scales time, effectively scaling the velocity, according to the
 * velocity profile
 ***/
double
apply_velocity_profile(struct velocity_profile *vp, double tx, double t0)
{
  double ti[vp->size];
  int n = 0;
  ti[0] = t0;
  for(int i=1; i<vp->size; i++) {
    ti[i] = (vp->velocities[i-1])*(vp->times[i] - vp->times[i-1]) + ti[i-1];
    if(tx > ti[i])
      n++;
  }

  return (1.0/vp->velocities[n])*(tx - ti[n]) + vp->times[n];
}

/***
 * Reads a file and returns a quarantine ratio profile
 * structure. Converts each line to two floats (time and rQ) stopping
 * after the first failure to convert.
 ***/
struct rQ_profile *
load_rQ_profile(char fname[])
{
  FILE *fp = uopen(fname, "r");
  int nlines = 0;
  while(!feof(fp)) {    
    char line[1024];
    fgets(line, 1024, fp);
    nlines++;
    double x, y;
    int n = sscanf(line, "%lf %lf", &x, &y);
    if(n != 2)
      break;
  }
  nlines--;
  rewind(fp);
  struct rQ_profile *rq = ualloc(sizeof(struct rQ_profile));
  rq->size = nlines;
  rq->times = ualloc(nlines*sizeof(double));
  rq->quarantine_ratios = ualloc(nlines*sizeof(double));
  for(int i=0; i<nlines; i++) {
    char line[1024];
    fgets(line, 1024, fp);
    sscanf(line, "%lf %lf", &rq->times[i], &rq->quarantine_ratios[i]);
  }
  fclose(fp);
  return rq;
}

/***
 * Determine and return rQ for given time
 ***/
double
get_rQ(struct rQ_profile *rq, double t)
{
  for(int i=0; i<rq->size-1; i++) {
    if(t < rq->times[i+1])
      return rq->quarantine_ratios[i];
  }
  return rq->quarantine_ratios[rq->size-1];
}

/***
 * Reads a file and returns an probability profile structure. Each
 * line begins with a time, then any number of items, where each item
 * is two floats separated by a colun ":". The first float in each
 * item is the infect propability and the second float the proportion
 * of the population that infect probability applies to.
 ***/
struct probability_profile *
load_probability_profile(char fname[])
{
  FILE *fp = uopen(fname, "r");
  int nlines = 0;
  while(!feof(fp)) {    
    char line[1024];
    fgets(line, 1024, fp);
    const char sep[] = " ";
    char *token = strtok(line, sep);
    int ncol = token != NULL ? 1 : 0;
    while(token != NULL) {
      token = strtok(NULL, sep);
      ncol++;
    }
    if(ncol == 0) {
      break;
    }
    nlines++;
  }
  nlines--;
  if(nlines <= 0) {
    fprintf(stderr, " Error parsing %s. Empty file?\n", fname);
    exit(-3);
  }
  rewind(fp);  
  struct probability_profile *pp = ualloc(sizeof(struct probability_profile));
  pp->n_times = nlines;
  pp->times = ualloc(sizeof(double)*nlines);
  pp->n_probs = ualloc(sizeof(int)*nlines);
  pp->population = ualloc(sizeof(double *)*nlines);
  pp->probability = ualloc(sizeof(double *)*nlines);
  for(int i=0; i<pp->n_times; i++) {
    char line[1024];
    fgets(line, 1024, fp);
    const char sep[] = " ";
    char *token = strtok(line, sep);
    pp->times[i] = strtod(token, NULL);
    int ncol = 0;
    while(token != NULL) {
      token = strtok(NULL, sep);
      ncol++;
    }
    pp->n_probs[i] = ncol-1;
  }
  for(int i=0; i<pp->n_times; i++) {
    if(pp->n_probs[i] == 0) {
      fprintf(stderr, " Error parsing %s. Malformed line: %d?\n", fname, i);
      exit(-3);
    }
  }
  rewind(fp);
  for(int i=0; i<pp->n_times; i++) {
    char line[1024];
    fgets(line, 1024, fp);
    const char sep[] = " ";
    char *token = strtok(line, sep);
    int ncol = pp->n_probs[i];
    pp->population[i] = ualloc(sizeof(double)*ncol);
    pp->probability[i] = ualloc(sizeof(double)*ncol);
    for(int j=0; j<ncol; j++) {
      token = strtok(NULL, sep);
      char *left = strsep(&token, ":");
      if(left == NULL) {
	fprintf(stderr, " Error parsing %s. Malformed line: %d, column: %d?\n", fname, i, j+1);
	exit(-3);
      }
      pp->probability[i][j] = strtod(left, NULL);
      char *right = strsep(&token, ":");
      if(right == NULL) {
	fprintf(stderr, " Error parsing %s. Malformed line: %d, column: %d?\n", fname, i, j+1);
	exit(-3);
      }
      pp->population[i][j] = strtod(right, NULL);
    }
  }
  fclose(fp);
  for(int i=0; i<pp->n_times; i++) {
    double sum = 0;
    for(int j=0; j<pp->n_probs[i]; j++) {
      sum += pp->population[i][j];
    }
    if(sum != 1) {
      fprintf(stderr, " Error on line: %d of %s, population fractions should sum to 1\n", i, fname);
      exit(-3);
    }
  }

  return pp;
}

/***
 * Determine infect probability given time and particle IDs
 ***/
double
get_infect_probability(struct probability_profile *pp, double t, int pid)
{
  double *pop, *pro;
  int i = 0;
  for(i=0; i<pp->n_times-1; i++) {
    if(t < pp->times[i+1]) {
      break;
    }
  }
  pop = pp->population[i];
  pro = pp->probability[i];
  double group = (double)pid / (double)N;
  double sum = 0;
  int j = 0;
  for(j=0; j<pp->n_probs[i]; j++) {
    sum += pop[j];
    if(group <= sum)
      break;
  }
  return pro[j];
}

/***
 * Return a random number in [0, 1)
 ***/
double
randr()
{
  double r = (double)random()/(double)RANDOM_MAX;
  return r;
}

/***
 * Return a random particle number, i.e. a random integer in [0, N)
 ***/
unsigned long int
random_particle()
{
  double r = randr();
  return (int)(r*N);
}

/***
 * Initialize the states array (*arr) with all particles to "S" except
 * for n_infected number of infectious ("I") and n_quarantined number
 * of quarantined ("Q")
 ***/
void
init_states(int *arr, int n_infected, int n_quarantined)
{
  for(int i=0; i<N; i++) {
    arr[i] = S;
  }

  for(int i=0; i<n_infected; i++) {
    int j = random_particle();
    arr[j] = I;
  }

  for(int j=random_particle(),i=0; i<n_quarantined;j=random_particle()) {
    if(arr[j] != I) {
      i++;
      arr[j] = Q;
    }
  }
  return;
}

/***
 * Initialize the N-length array *arr to t0. Used to track the time of
 * infection.
 ***/
void
init_I_t0(double *arr, double t0)
{
  for(int i=0; i<N; i++) {
    arr[i] = t0;
  }
  return;
}

/***
 * Initialize the N-length integer array *Rt to zero. *Rt is used to
 * track the number of transmissions, in turn to be used to obtain the
 * basic reproduction number.
 ***/
void
init_R(int *Rt)
{
  for(int i=0; i<N; i++) {
    Rt[i] = 0;
  }
  return;
}

/***
 * Go over state array *arr and based on exposed_time and
 * infectious_time transition "E"s to "I"s or "Q"s or "I"s and "Q"s to
 * "RI"s and "RQ"s.
 ***/
void
update(int *arr, double *times, double t, double quarantine_ratio, double exposed_time, double infectious_time)
{
  for(int i=0; i<N; i++) {
    int xi = t - times[i] > infectious_time;
    int xj = t - times[i] > exposed_time;
    {
      int y = arr[i] == E;
      if(xj & y) {
	/* Turn an exposed ("E") to either "Q" or "I" */
	if(randr() < quarantine_ratio) {
	  arr[i] = Q;
	} else {
	  arr[i] = I;
	}
      }
    }
    {
      int y = arr[i] == I;
      /* Recover an infectious */
      if(xi & y) {
	arr[i] = RI;
      }
    }
    {
      int y = arr[i] == Q;
      /* Recover a quaratnined */
      if(xi & y) {
	arr[i] = RQ;
      }
    }
  }
  return;
}

/***
 * Print cycles, time, average reproduction number (over all recovered
 * individuals), and the number of individuals in each state.
 ***/
void
print_state(int *arr, int *Rt, double t, int cycles)
{
  int nR = 0;
  double Rmean = 0;
  for(int i=0; i<N; i++)
    if(arr[i] == RI) {
      Rmean += Rt[i];
      nR++;
    }
  if(nR == 0) {
    Rmean = 0;
  } else {
    Rmean /= (double)nR;
  }

  printf(" %d %lf %lf ", cycles, t, Rmean);
  int counts[NSTATES];
  for(int i=0; i<NSTATES; i++)
    counts[i] = 0;
  
  for(int i=0; i<N; i++)
    counts[arr[i]] += 1;
  
  for(int i=0; i<NSTATES; i++)
    printf(" %10d", counts[i]);
  
  printf("\n");
}

static char doc[] =
  "Run SIR over collisions file FILE_NAME for N particles";

static char args_doc[] = "N FILE_NAME";

static struct argp_option options[] =
  {
   {"infectious-time",     't', "T",      0,  "Time each infected is infectious, in simulation time units" },
   {"exposed-time",        'e', "T",      0,  "Exposed time (infected but not infectious yet), in simulation units" },
   {"period",              'p', "P",      0,  "Period: restart every P time units" },
   {"quarantine-ratio",    'r', "R",      0,  "Ratio of infected to quarantined" },
   {"infect-probability",  'i', "R",      0,  "Probability a collision will result in infection" },
   {"initial-infectious",  'I', "NI",     0,  "Initial number of infectious" },
   {"initial-quarantined", 'Q', "NQ",     0,  "Initial number of quarantined" },
   {"random-seed",         's', "S",      0,  "Use S to initialize the random number generator" },
   {"velocity-scaling",    'v', "V",      0,  "Scale velocity by a factor of V" },
   {"velocity-profile",    'f', "F",      0,  "Scale velocities according to profile in file F" },
   {"quarantine-profile",  'q', "F",      0,  "Change quarantined ratio according to profile in file F" },
   {"probability-profile", 'b', "F",      0,  "Change infection probability according to profile in file F" },
   { 0 }
  };

struct arguments
{
  char *fname;
  char *velocity_profile, *quarantine_profile, *probability_profile;
  int N;
  double infectious_time, exposed_time, period, quarantine_ratio, velocity_scaling, infect_probability;
  int initial_infected, initial_quarantined;
  unsigned long int seed;
};

static bool opts_parsed[3][2] = {{false, false},
				 {false, false},
				 {false, false},};

static error_t
parse_opt(int key, char *arg, struct argp_state *state)
{
  struct arguments *arguments = state->input;
  
  switch (key) {
  case 't':
    arguments->infectious_time = strtod(arg, NULL);
    break;
  case 'e':
    arguments->exposed_time = strtod(arg, NULL);
    break;
  case 'p':
    arguments->period = strtod(arg, NULL);
    break;
  case 'v':
    arguments->velocity_scaling = strtod(arg, NULL);
    opts_parsed[1][0] = true;
    break;
  case 's':
    arguments->seed = strtoul(arg, NULL, 10);
    break;
  case 'r':
    arguments->quarantine_ratio = strtod(arg, NULL);
    if(arguments->quarantine_ratio > 1 || arguments->quarantine_ratio < 0) {
      argp_failure(state, 1, 0, " -%c :option argument should be within [0,1]\n", key);
      return 1;
    }
    opts_parsed[0][0] = true;
    break;
  case 'i':
    arguments->infect_probability = strtod(arg, NULL);
    if(arguments->infect_probability > 1 || arguments->infect_probability < 0) {
      argp_failure(state, 1, 0, " -%c :option argument should be within [0,1]\n", key);
      return 1;
    }
    opts_parsed[2][0] = true;    
    break;
  case 'I':
    arguments->initial_infected = strtoul(arg, NULL, 10);
    break;
  case 'Q':
    arguments->initial_quarantined = strtoul(arg, NULL, 10);
    break;
  case 'f':
    arguments->velocity_profile = arg;
    opts_parsed[1][1] = true;
    break;
  case 'b':
    arguments->probability_profile = arg;
    opts_parsed[2][1] = true;
    break;
  case 'q':
    arguments->quarantine_profile = arg;
    opts_parsed[0][1] = true;
    break;
  case ARGP_KEY_ARG:
    if (state->arg_num >= 2)
      /* Too many arguments. */
      argp_usage(state);
    if(state->arg_num == 0)
      arguments->N = strtoul(arg, NULL, 10);
    if(state->arg_num == 1)
      arguments->fname = arg;
    if(opts_parsed[0][0] && opts_parsed[0][1])
      argp_failure(state, 1, 0, " -%c and -%c cannot be used together\n", 'r', 'q');
    if(opts_parsed[1][0] && opts_parsed[1][1])
      argp_failure(state, 1, 0, " -%c and -%c cannot be used together\n", 'v', 'f');
    if(opts_parsed[2][0] && opts_parsed[2][1])
      argp_failure(state, 1, 0, " -%c and -%c cannot be used together\n", 'b', 'i');
    break;
  case ARGP_KEY_END:
    if (state->arg_num < 2)
      /* Not enough arguments. */
      argp_usage(state);
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

int
main(int argc, char *argv[])
{
  struct arguments arguments;
  arguments.infectious_time = 3;
  arguments.exposed_time = 0;
  arguments.period = 15;
  arguments.quarantine_ratio = 0;
  arguments.velocity_scaling = 1;
  arguments.initial_infected = 1;
  arguments.initial_quarantined = 0;
  arguments.infect_probability = 1;
  arguments.seed = 7;
  arguments.velocity_profile = NULL;
  arguments.quarantine_profile = NULL;
  arguments.probability_profile = NULL;
 
  argp_parse(&argp, argc, argv, 0, 0, &arguments);
  
  unsigned long int seed = arguments.seed;
  double infectious_time = arguments.infectious_time;
  double exposed_time = arguments.exposed_time;
  double period = arguments.period;
  double quarantine_ratio = arguments.quarantine_ratio;
  double velocity_scaling = arguments.velocity_scaling;
  int initial_infected = arguments.initial_infected;
  int initial_quarantined = arguments.initial_quarantined;
  double infect_probability = arguments.infect_probability;
  N = arguments.N;
  char *fname = arguments.fname;
  char *velocity_profile = arguments.velocity_profile;
  char *quarantine_profile = arguments.quarantine_profile;
  char *probability_profile = arguments.probability_profile;

  struct velocity_profile *vp = NULL;
  if(velocity_profile != NULL)
    vp = load_velocity_profile(velocity_profile);

  struct rQ_profile *rq = NULL;
  if(quarantine_profile != NULL)
    rq = load_rQ_profile(quarantine_profile);

  struct probability_profile *pp = NULL;
  if(probability_profile != NULL)
    pp = load_probability_profile(probability_profile);

  int *state_arr = ualloc(sizeof(int)*N);
  double *I_t0 = ualloc(sizeof(double)*N);
  int *Rt = ualloc(sizeof(double)*N);
  FILE *fp = uopen(fname, "r");
  /* Read first line, get t0, then rewind */
  double t0 = 0;
  {
    char line[1024];
    fgets(line, 1024, fp);
    unsigned long int p0, p1;
    sscanf(line, "%lf %lu %lu", &t0, &p0, &p1);
    rewind(fp);
  }
  
  srandom(seed);

  /* Iterate line-by-line. Restart the SIR model every "period" time
     units, incrementing "cycles" each time */
  int cycles = 0;
  for(int iter=0; !feof(fp); iter++) {
    char line[1024];
    fgets(line, 1024, fp);
    unsigned long int p0, p1;
    double tx, t;
    sscanf(line, "%lf %lu %lu", &tx, &p0, &p1);
    /* Scaling the velocity is equivalent to scaling time */
    if(vp == NULL) {
      t = (tx-t0)/velocity_scaling;
    } else {
      t = apply_velocity_profile(vp, tx, t0);
    }
    if(t >= period || cycles == 0) {
      t0 = tx;
      init_states(state_arr, initial_infected, initial_quarantined);
      init_I_t0(I_t0, 0);
      init_R(Rt);
      cycles += 1;
      print_state(state_arr, Rt, 0, cycles-1);
    }

    if(rq != NULL)
      quarantine_ratio = get_rQ(rq, t);
    
    int pij[2] = {p0, p1};
    /* Update states if one of the collision participants is exposed
     * since they may be infectious by now */
    for(int i=0; i<2; i++)
      if(pij[i] == E)
	update(state_arr, I_t0, t, quarantine_ratio, exposed_time, infectious_time);
    /* Will be set to 1 in case of an "Infection event", i.e. if a
       collision between an "S" and an "I" happens */
    int xy = 0; 
    for(int i=0; i<2; i++) {
      int pi = pij[(i+0) % 2];
      int pj = pij[(i+1) % 2];
      int x = state_arr[pi] == I;
      int y = state_arr[pj] == S;
      if(x & y) {
    	double dt = t - I_t0[pi];
    	if(dt < infectious_time) {
	  if(pp != NULL) {
	    // Determine probability according to susceptible
	    infect_probability = get_infect_probability(pp, t, pj);
	  }
	  if(randr() < infect_probability) {
	    state_arr[pj] = E;
	    Rt[pi] += 1;
	    I_t0[pj] = t;
	  }
    	}
      }
      xy += (x & y);
    }

    /* Go over state and recover, then print state. This should only
       be necessary if a collision event occured. */
    if(xy) {
      update(state_arr, I_t0, t, quarantine_ratio, exposed_time, infectious_time);
      print_state(state_arr, Rt, t, cycles-1);
    }
  }

  if(vp != NULL)
    free(vp);
  if(rq != NULL)
    free(rq);
  free(state_arr);
  free(Rt);
  free(I_t0);
  fclose(fp);
  return 0;
}
