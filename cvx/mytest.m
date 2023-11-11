function [solved,min_value] = mytest(t_n, a_n, alpha, beta, N, m)
cvx_begin quiet
    variable q_n(N);
    expression u(N);
    fprintf("Calculating...")
    for i = 1:N
        u(i)=[rel_entr(alpha.*q_n(i).*m,1)+rel_entr(1,N.*beta.*q_n(i)-pow_pos(a_n(i).*N,2))];
    end
    minimize(sum(u(:)))
    subject to
        pow_pos(a_n,2)*N*inv_pos(beta) <= q_n;
        q_n >= 0;
        q_n <= 1;
        m == sum(q_n.*t_n);
    cvx_end

    solved = q_n;
    min_value = cvx_optval; % 获取目标函数的最小值
    fprintf('Minimum value: %f\n', min_value);
end