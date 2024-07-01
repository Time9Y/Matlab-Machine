function Y_hat = regRF_predict(p_train, model)
    % requires 2 arguments
    % p_train: data matrix
    % model: generated via regRF_train function

	if nargin ~= 2
		error('need atleast 2 parameters, X matrix and model');
	end
	
	Y_hat = mexRF_predict(p_train', model.lDau, model.rDau, model.nodestatus, model.nrnodes, ...
        model.upper, model.avnode, model.mbest, model.ndtree, model.ntree);
    
    if ~isempty(find(model.coef, 1)) % for bias corr
        Y_hat = model.coef(1) + model.coef(2) * Y_hat;
    end

	clear mexRF_predict