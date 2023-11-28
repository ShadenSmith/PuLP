/*
//@HEADER
// *****************************************************************************
//
// PULP: Multi-Objective Multi-Constraint Partitioning Using Label Propagation
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact  George M. Slota (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//
// *****************************************************************************
//@HEADER
*/

#include <cassert>

using namespace std;

extern int64_t seed;

/*
'########::'########:::'#######::'########::
 ##.... ##: ##.... ##:'##.... ##: ##.... ##:
 ##:::: ##: ##:::: ##: ##:::: ##: ##:::: ##:
 ########:: ########:: ##:::: ##: ########::
 ##.....::: ##.. ##::: ##:::: ##: ##.....:::
 ##:::::::: ##::. ##:: ##:::: ##: ##::::::::
 ##:::::::: ##:::. ##:. #######:: ##::::::::
..:::::::::..:::::..:::.......:::..:::::::::
*/
int64_t* label_prop(pulp_graph_t& g, int64_t num_parts, int64_t* parts,
  int64_t label_prop_iter, double balance_vert_lower)
{
  int64_t num_verts = g.n;
  int64_t* part_sizes = new int64_t[num_parts];
  for (int64_t i = 0; i < num_parts; ++i)
    part_sizes[i] = 0;

  int64_t num_changes;
  int64_t* queue = new int64_t[num_verts*QUEUE_MULTIPLIER];
  int64_t* queue_next = new int64_t[num_verts*QUEUE_MULTIPLIER];
  bool* in_queue = new bool[num_verts];
  bool* in_queue_next = new bool[num_verts];
  int64_t queue_size = num_verts;
  int64_t next_size = 0;

  double avg_size = (double)num_verts / (double)num_parts;
  double min_size = avg_size * balance_vert_lower;

#pragma omp parallel
{
  xs1024star_t xs;
  xs1024star_seed((uint64_t)(seed + omp_get_thread_num()), &xs);

#pragma omp for
  for (int64_t i = 0; i < num_verts; ++i)
    parts[i] = (int64_t)((uint64_t)xs1024star_next(&xs) % (uint64_t)num_parts);
  
  int64_t* part_sizes_thread = new int64_t[num_parts];
  for (int64_t i = 0; i < num_parts; ++i) 
    part_sizes_thread[i] = 0;

#pragma omp for schedule(static) nowait
  for (int64_t i = 0; i < num_verts; ++i)
      ++part_sizes_thread[parts[i]];

  for (int64_t i = 0; i < num_parts; ++i) 
#pragma omp atomic
    part_sizes[i] += part_sizes_thread[i];

  //delete [] part_sizes_thread;


#pragma omp for schedule(static) nowait
  for (int64_t i = 0; i < num_verts; ++i)
    queue[i] = i;
#pragma omp for schedule(static)
  for (int64_t i = 0; i < num_verts; ++i)
    in_queue_next[i] = false;

  int64_t* part_counts = new int64_t[num_parts];
  int64_t thread_queue[ THREAD_QUEUE_SIZE ];
  int64_t thread_queue_size = 0;
  int64_t thread_start;

  for (int64_t num_iter = 0; num_iter < label_prop_iter; ++num_iter)
  { 
    num_changes = 0;

#pragma omp for schedule(guided) reduction(+:num_changes)
    for (int64_t i = 0; i < queue_size; ++i)
    {
      int64_t v = queue[i];
      in_queue[v] = false;
      for (int64_t j = 0; j < num_parts; ++j)
        part_counts[j] = 0;

      uint64_t v_degree = out_degree(g, v);
      int64_t* outs = out_vertices(g, v);
      for (uint64_t j = 0; j < v_degree; ++j)
      {
        int64_t out = outs[j];
        int64_t part = parts[out];
        part_counts[part] += out_degree(g, out);
      }
      
      int64_t part = parts[v];
      int64_t max_count = -1;
      int64_t max_part = -1;
      int64_t num_max = 0;
      for (int64_t p = 0; p < num_parts; ++p)
      {
        if (part_counts[p] == max_count && 
            (part_sizes[p]-1) > (int64_t)min_size)
        {
          part_counts[num_max++] = p;
        }
        else if (part_counts[p] > max_count && 
                 (part_sizes[p]-1) > (int64_t)min_size)
        {
          max_count = part_counts[p];
          max_part = p;
          num_max = 0;
          part_counts[num_max++] = p;
        }
      }

      if (num_max > 1) {
        int64_t rand_val;

        // TODO: thread-local RNG
        #pragma omp critical
        {
          rand_val = (int64_t)rand();
        }

        max_part = part_counts[rand_val % num_max];
      }

      if (max_part != part && 
          (part_sizes[part]-1) > (int64_t)min_size)
      {
    #pragma omp atomic
        ++part_sizes[max_part];
    #pragma omp atomic
        --part_sizes[part];
        
        parts[v] = max_part;
        ++num_changes;

        if (!in_queue_next[v])
        {
          in_queue_next[v] = true;
          thread_queue[thread_queue_size++] = v;

          if (thread_queue_size == THREAD_QUEUE_SIZE)
          {
#pragma omp atomic capture
            thread_start = next_size += thread_queue_size;
            
            thread_start -= thread_queue_size;
            for (int64_t l = 0; l < thread_queue_size; ++l)
              queue_next[thread_start+l] = thread_queue[l];
            thread_queue_size = 0;
          }
        }
        for (uint64_t j = 0; j < v_degree; ++j)
        {
          if (!in_queue_next[outs[j]])
          {
            in_queue_next[outs[j]] = true;
            thread_queue[thread_queue_size++] = outs[j];

            if (thread_queue_size == THREAD_QUEUE_SIZE)
            {
#pragma omp atomic capture
              thread_start = next_size += thread_queue_size;
              
              thread_start -= thread_queue_size;
              for (int64_t l = 0; l < thread_queue_size; ++l)
                queue_next[thread_start+l] = thread_queue[l];
              thread_queue_size = 0;
            }
          }
        }
      }
    }
#pragma omp atomic capture
    thread_start = next_size += thread_queue_size;
    
    thread_start -= thread_queue_size;
    for (int64_t l = 0; l < thread_queue_size; ++l)
      queue_next[thread_start+l] = thread_queue[l];
    thread_queue_size = 0;

#pragma omp barrier
    
    ++num_iter;
#pragma omp single
{
#if VERBOSE
    printf("%d\n", next_size);
#endif

    int64_t* temp = queue;
    queue = queue_next;
    queue_next = temp;
    bool* temp_b = in_queue;
    in_queue = in_queue_next;
    in_queue_next = temp_b;

    queue_size = next_size;
    next_size = 0;

#if OUTPUT_STEP
  evaluate_quality(g, num_parts, parts);
#endif
} // end single
  } // end while

  delete [] part_counts;
} // end parallel

  delete [] queue;
  delete [] queue_next;
  delete [] in_queue;
  delete [] in_queue_next;

  return parts;
}


int64_t* label_prop_weighted(pulp_graph_t& g, int64_t num_parts, int64_t* parts,
  int64_t label_prop_iter, double balance_vert_lower)
{
  int64_t num_verts = g.n;  
  bool has_vwgts = (g.vertex_weights != NULL);
  bool has_ewgts = (g.edge_weights != NULL);
  if (!has_vwgts) g.vertex_weights_sum = g.n;

  int64_t* part_sizes = new int64_t[num_parts];
  for (int64_t i = 0; i < num_parts; ++i)
    part_sizes[i] = 0;

  int64_t num_changes;
  int64_t* queue = new int64_t[num_verts*QUEUE_MULTIPLIER];
  int64_t* queue_next = new int64_t[num_verts*QUEUE_MULTIPLIER];
  bool* in_queue = new bool[num_verts];
  bool* in_queue_next = new bool[num_verts];
  int64_t queue_size = num_verts;
  int64_t next_size = 0;

  double avg_size = (double)g.vertex_weights_sum / (double)num_parts;
  double min_size = avg_size * balance_vert_lower;

#pragma omp parallel
{
  xs1024star_t xs;
  xs1024star_seed((uint64_t)(seed + omp_get_thread_num()), &xs);

#pragma omp for
  for (int64_t i = 0; i < num_verts; ++i) {
    parts[i] = (int64_t)(xs1024star_next(&xs) % (uint64_t)num_parts);
  }

  int64_t* part_sizes_thread = new int64_t[num_parts];
  for (int64_t i = 0; i < num_parts; ++i) 
    part_sizes_thread[i] = 0;

#pragma omp for schedule(static) nowait
  for (int64_t i = 0; i < num_verts; ++i)
    if (has_vwgts)
      part_sizes_thread[parts[i]] += g.vertex_weights[i];
    else 
      ++part_sizes_thread[parts[i]];

  for (int64_t i = 0; i < num_parts; ++i) 
#pragma omp atomic
    part_sizes[i] += part_sizes_thread[i];

  delete [] part_sizes_thread;

#pragma omp for schedule(static) nowait
  for (int64_t i = 0; i < num_verts; ++i)
    queue[i] = i;
#pragma omp for schedule(static)
  for (int64_t i = 0; i < num_verts; ++i)
    in_queue_next[i] = false;

  int64_t* part_counts = new int64_t[num_parts];
  int64_t thread_queue[ THREAD_QUEUE_SIZE ];
  int64_t thread_queue_size = 0;
  int64_t thread_start;

  for (int64_t num_iter = 0; num_iter < label_prop_iter; ++num_iter)
  { 
    num_changes = 0;

#pragma omp for schedule(guided) reduction(+:num_changes)
    for (int64_t i = 0; i < queue_size; ++i)
    {
      int64_t v = queue[i];
      int64_t v_weight = 1;
      if (has_vwgts) v_weight = g.vertex_weights[v];

      in_queue[v] = false;
      for (int64_t j = 0; j < num_parts; ++j)
        part_counts[j] = 0;

      uint64_t v_degree = out_degree(g, v);
      int64_t* outs = out_vertices(g, v);
      int64_t* weights = out_weights(g, v);
      for (uint64_t j = 0; j < v_degree; ++j)
      {
        int64_t out = outs[j];
        int64_t part_out = parts[out];
        double weight_out = 1.0;
        if (has_ewgts) weight_out = (double)weights[j];

        if (out >= g.n) printf("invalid out: %lld, n: %lld\n", out, g.n);
        if (out < 0) printf("invalid out: %lld\n", out);
        if (part_out >= num_parts) {
          printf("invalid part_out: %lld, num_parts: %lld\n", part_out, num_parts);
          fflush(stdout);
          exit(1);
        }
        part_counts[part_out] += ((double)out_degree(g, out))*weight_out;
      }
      
      int64_t part = parts[v];
      int64_t max_count = -1;
      int64_t max_part = -1;
      int64_t num_max = 0;
      for (int64_t p = 0; p < num_parts; ++p)
      {
        if (part_counts[p] == max_count)
        {
          part_counts[num_max++] = p;
        }
        else if (part_counts[p] > max_count)
        {
          max_count = part_counts[p];
          max_part = p;
          num_max = 0;
          part_counts[num_max++] = p;
        }
      }

      if (num_max > 1) {
          max_part = part_counts[(xs1024star_next(&xs) % num_max)];
      }

      if (max_part != part && 
          (part_sizes[part] - v_weight > (int64_t)min_size))
      {
    #pragma omp atomic
        part_sizes[max_part] += v_weight;
    #pragma omp atomic
        part_sizes[part] -= v_weight;
        
        if(max_part >= num_parts) {
          printf("BAD max_part: %lld, num_parts: %lld\n", max_part, num_parts);
          fflush(stdout);
          exit(1);
        }
        parts[v] = max_part;
        ++num_changes;

        if (!in_queue_next[v])
        {
          in_queue_next[v] = true;
          thread_queue[thread_queue_size++] = v;

          if (thread_queue_size == THREAD_QUEUE_SIZE)
          {
#pragma omp atomic capture
            thread_start = next_size += thread_queue_size;
            
            thread_start -= thread_queue_size;
            for (int64_t l = 0; l < thread_queue_size; ++l)
              queue_next[thread_start+l] = thread_queue[l];
            thread_queue_size = 0;
          }
        }
        for (uint64_t j = 0; j < v_degree; ++j)
        {
          if (!in_queue_next[outs[j]])
          {
            in_queue_next[outs[j]] = true;
            thread_queue[thread_queue_size++] = outs[j];

            if (thread_queue_size == THREAD_QUEUE_SIZE)
            {
#pragma omp atomic capture
              thread_start = next_size += thread_queue_size;
              
              thread_start -= thread_queue_size;
              for (int64_t l = 0; l < thread_queue_size; ++l)
                queue_next[thread_start+l] = thread_queue[l];
              thread_queue_size = 0;
            }
          }
        }
      }
    }
#pragma omp atomic capture
    thread_start = next_size += thread_queue_size;
    
    thread_start -= thread_queue_size;
    for (int64_t l = 0; l < thread_queue_size; ++l)
      queue_next[thread_start+l] = thread_queue[l];
    thread_queue_size = 0;

#pragma omp barrier
    
    ++num_iter;
#pragma omp single
{
#if VERBOSE
    printf("%d\n", next_size);
#endif

    int64_t* temp = queue;
    queue = queue_next;
    queue_next = temp;
    bool* temp_b = in_queue;
    in_queue = in_queue_next;
    in_queue_next = temp_b;

    queue_size = next_size;
    next_size = 0;

#if OUTPUT_STEP
  evaluate_quality(g, num_parts, parts);
#endif
} // end single
  } // end while

  delete [] part_counts;
} // end parallel

  delete [] queue;
  delete [] queue_next;
  delete [] in_queue;
  delete [] in_queue_next;

  return parts;
}
