#include <TinyAD/Scalar.hh>
#include <iostream>
#include <cmath>

struct NewtonSolverSettings
{
    int max_iterations = 1000;
    double tol = 1e-9;
    // Armijo line search parameters
    double gamma = 0.1;
    double beta = 0.8;
    double t_max = 1.0;
};

template<int N, int M>  // N = number of variables, M = number of equations
class NewtonSolver
{   
    public:
        using AD_N = TinyAD::Double<N>;
        using FuncSig = Eigen::Matrix<AD_N, M, 1>(*)(const Eigen::Matrix<AD_N, N, 1>&, Eigen::Matrix<AD_N, N, 1>&);
        using FuncSigDouble = Eigen::Matrix<double, M, 1>(*)(const Eigen::Matrix<double, N, 1>&, Eigen::Matrix<double, N, 1>&);

        bool get_success() const
        {
            return success;
        }

        int get_iterations() const
        {
            return iterations;
        }
        
        void set_settings(int max_iterations, double tol)
        {
            settings.max_iterations = max_iterations;
            settings.tol = tol;
        }

        NewtonSolverSettings get_settings() const
        {
            return settings;
        }

        void set_initial_guess(Eigen::Matrix<double, N, 1>& guess)
        {
            this->initial_guess = &guess; // no new allocation
        }

        void print_initial_guess() const
        {
            Eigen::Matrix<double, N, 1> vals;
            for (int i = 0; i < N; ++i)
                vals[i] = initial_guess[i].val;
            std::cout << "Initial guess values: " << vals.transpose() << std::endl;
        }

        void Jacobian(
            const Eigen::Matrix<AD_N, N, 1> &input,
            FuncSig func)
        {   
            if (this->J == nullptr) {
                this->J = new Eigen::Matrix<double, M, N>(); // allocate once, no copy
            }
            if (this->F_val_ad == nullptr) {
                this->F_val_ad = new Eigen::Matrix<AD_N, M, 1>(); // allocate once, no copy
            }
            
            func(input, *F_val_ad); // compute F_val_ad at the current input

            for (int i = 0; i < M; ++i)
                this->J->row(i) = (*F_val_ad)[i].grad.transpose(); // write directly
        }

        const Eigen::Ref<const Eigen::MatrixXd> get_solution() const
        {
            return solution;
        }


    protected:
        
        NewtonSolverSettings settings;
        bool success = false;
        int iterations = 0;
        Eigen::Vector<double, N> solution = Eigen::Matrix<double, N, 1>::Zero();
        Eigen::Matrix<double, N, 1>* initial_guess = new Eigen::Matrix<double, N, 1>(Eigen::Matrix<double, N, 1>::Zero());        
        Eigen::Matrix<double, M, N>* J = nullptr;
        Eigen::Matrix<double, M, 1>* F_val = nullptr; 
        Eigen::Matrix<AD_N, M, 1>* F_val_ad = nullptr;              
        Eigen::Matrix<double, N, 1> d = Eigen::Matrix<double, N, 1>::Zero(); // descent direction

        // Armijo preallocated vectors
        Eigen::Matrix<double, M, 1>* lhs_armijo = nullptr;
        Eigen::Matrix<double, M, 1>* rhs_armijo = nullptr;

    protected:
        void assign_F0(const Eigen::Matrix<AD_N, M, 1> & F0_active) 
            {   
                if (F_val == nullptr)
                {
                    F_val = new Eigen::Matrix<double, M, 1>(); // allocate once, no copy
                }
                for (int i = 0; i < M; ++i)
                    (*F_val)[i] = F0_active[i].val;
            }

        template<int Size>
        void assign_AD_vector(
            const Eigen::Matrix<AD_N, Size, 1>& vec_source,
            Eigen::Matrix<double, Size, 1>*& vec_dest)
        {
            if (vec_dest == nullptr)
                vec_dest = new Eigen::Matrix<double, Size, 1>();

            for (int i = 0; i < Size; ++i)
                (*vec_dest)[i] = vec_source[i].val;
        }

        void assign_solution(const Eigen::Matrix<AD_N, N, 1> & sol) 
        {   
            for (int i = 0; i < N; ++i)
                this->solution[i] = sol[i].val;
        }

        bool multivariate_check_minor(Eigen::Matrix<double, M, 1> & lhs , Eigen::Matrix<double, M, 1> & rhs) 
        {
            for (int i = 0; i < M; ++i) {
                if (lhs[i] > rhs[i]) {
                    return false;
                } 
            }
            return true; 
        }

        double merit_fun(const Eigen::Matrix<double, N, 1> & x, Eigen::Matrix<double, M, 1> *& tmp,  FuncSigDouble func) {
            func(x, *tmp);
            return 0.5*pow((*tmp).norm(),2);
        }

        double armijo_search(Eigen::Matrix<double, N, 1> & x, Eigen::Vector<double, N> & d, FuncSigDouble func, Eigen::Vector<double, N> & f_val_x ) {
            if (lhs_armijo == nullptr)
                lhs_armijo = new Eigen::Matrix<double, M, 1>();
            if (rhs_armijo == nullptr)
                rhs_armijo = new Eigen::Matrix<double, M, 1>();
            int iter = 0;
            double t = settings.t_max;
            Eigen::Matrix<double, N,1> merit_grad = (*J).transpose() * (f_val_x);
            // while(!multivariate_check_minor(lhs,rhs) && iter < 50) {
            while (!(merit_fun(x+t*d,lhs_armijo,func) - merit_fun(x,rhs_armijo,func) - settings.gamma*t*merit_grad.transpose()*d <= 0)) {
                t *= settings.beta;
                // assign_M_AD_vector(func(x + t*d),lhs_armijo); // the sum of AD vector and standarf vector gives AD vector;
                // assign_M_AD_vector(func(x) + settings.gamma * t * ((*J)*d) , rhs_armijo);
                std::cout << "Current t " << t << std::endl;
                iter += 1;
            }
            return t;
        }

    public:
        void solve(FuncSig func, FuncSigDouble func_double)
        {   
            Eigen::Matrix<AD_N, N, 1> x = AD_N::make_active(*initial_guess);
            // func(x, *F_val_ad); // compute F_val_ad at the initial guess
            // assign_F0(F0_active);
            if (F_val == nullptr)
                {
                    F_val = new Eigen::Matrix<double, M, 1>(); // allocate once, no copy
                }
            func_double(*initial_guess, *F_val); // compute F_val at the initial guess
            double norm_F = (*F_val).norm();
            Eigen::Matrix<double, N, 1>* x_double = nullptr;
            assign_AD_vector(x, x_double);  

            if (norm_F > settings.tol)
            {
                while (iterations < settings.max_iterations && !success)
                {
                    
                    Jacobian(x, func);
                    auto lu = J->fullPivLu();
                    d = -lu.solve(*F_val);

                    double t = armijo_search(*x_double,d,func_double,*F_val); 
                    // double t = 1.0; // for now, no line search
                    x += t*d;
                    assign_AD_vector(x, x_double); // update x_double for the Armijo line search in the next iteration
                    
                    std::cout << "t Armijo line search at step " << iterations << " has value " << t << std::endl; 
                    iterations++;
                    // update F0 at the new x
                    func(x, *F_val_ad);
                    assign_F0(*F_val_ad);
                    norm_F = (*F_val).norm();
                    if (norm_F < settings.tol)
                    {
                        success = true;
                        assign_solution(x);
                        std::cout << "Converged in " << iterations << " iterations." << std::endl;
                        return;
                    }
                }
            }
            else
            {
                success = true;
                assign_solution(*initial_guess);
            }
            
        }
};