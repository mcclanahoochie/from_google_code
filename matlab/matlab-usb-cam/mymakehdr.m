function hdr = mymakehdr(img0)
	%MAKEHDR    Create high dynamic range image.

	% Get the minimum exposure image from the user or make a first pass through
	% the images to find the lowest exposure image.
	[baseTime, baseFStop] = getAverageExposure();

	% Create output variables for an accumulator and the number of LDR images
	% that contributed to each pixel.
	meta = img0(:,:,:,1);

	[hdr, properlyExposedCount] = makeContainers(meta);

	someUnderExposed = false(size(hdr));
	someOverExposed = false(size(hdr));
	someProperlyExposed = false(size(hdr));

	% re-make options
	optionsrelexp = [.09 1 2];
	optionsminclip = 5;
	optionsmaxclip = 250;

	% Construct the HDR image by iterating over the LDR images.
	for p = 1:3

		relExposure = optionsrelexp(p);

		% Read the LDR image
		ldr = img0(:,:,:,p);
		
		underExposed = ldr < optionsminclip;
		someUnderExposed = someUnderExposed | underExposed;
		
		overExposed = ldr > optionsmaxclip;
		someOverExposed = someOverExposed | overExposed;
		
		properlyExposed = ~(underExposed | overExposed);
		someProperlyExposed = someProperlyExposed | properlyExposed;
		
		properlyExposedCount(properlyExposed) = properlyExposedCount(properlyExposed) + 1;
		
		% Remove over- and under-exposed values.
		ldr(~properlyExposed) = 0;
		
		% Bring the intensity of the LDR image into a common HDR domain by
		% "normalizing" using the relative exposure, and then add it to the
		% accumulator.
		hdr = hdr + single(ldr) ./ relExposure;
		
	end

	% Average the values in the accumulator by the number of LDR images that
	% contributed to each pixel to produce the HDR radiance map.
	hdr = hdr ./ max(properlyExposedCount, 1);

	% For pixels that were completely over-exposed, assign the maximum
	% value computed for the properly exposed pixels.
	hdr(someOverExposed & ~someUnderExposed & ~someProperlyExposed) = max(hdr(someProperlyExposed));

	% For pixels that were completely under-exposed, assign the
	% minimum value computed for the properly exposed pixels.
	hdr(someUnderExposed & ~someOverExposed & ~someProperlyExposed) = min(hdr(someProperlyExposed));

	% For pixels that were sometimes underexposed, sometimes
	% overexposed, and never properly exposed, use roifill.
	fillMask = imdilate(someUnderExposed & someOverExposed & ~someProperlyExposed, ones(3,3));
	if any(fillMask(:))
		hdr(:,:,1) = roifill(hdr(:,:,1), fillMask(:,:,1));
		hdr(:,:,2) = roifill(hdr(:,:,2), fillMask(:,:,2));
		hdr(:,:,3) = roifill(hdr(:,:,3), fillMask(:,:,3));
	end




function [baseTime, baseFStop] = getAverageExposure()
	% Extract the average exposure (assuming constant illumination) from a set
	% of files containing EXIF metadata.  The average exposure may not actually
	% correspond to the exposure of any particular image.

	minTime = 0.1;
	minFStop = 0.1;
	maxTime = 0.9;
	maxFStop = 0.9;

	% Determine the "middle" exposure value.  It's easier to manipulate
	% exposure time rather than f/stop.
	re = computeRelativeExposure(minFStop, minTime, ...
		                         maxFStop, maxTime);
	baseFStop = minFStop;
	baseTime  = minTime * log2(re);




function relExposure = computeRelativeExposure(f1, t1, f2, t2)
	% Exposure varies directly with the exposure time and inversely with the
	% square of the F-stop number. 
	relExposure = (f1 / f2)^2 * (t2 / t1);




function options = parseArgs(varargin)
	% Parse the parameter-value pairs, getting default values.
	knownParams = {'BaseFile',         'basefile', '', {'char'},    {'vector'};
		           'ExposureValues',   'expvals',  [], {'numeric'}, {'vector', 'real', 'finite', 'nonnan'};
		           'RelativeExposure', 'relexp',   [], {'numeric'}, {'vector', 'real', 'finite', 'positive', 'nonzero'};
		           'MinimumLimit',     'minclip',   5, {'numeric'}, {'scalar', 'integer', 'real', 'nonnan', 'positive'};
		           'MaximumLimit',     'maxclip', 250, {'numeric'}, {'scalar', 'integer', 'real', 'nonnan', 'positive'}};
	options = parseParameterValuePairs(mfilename, knownParams, varargin{:});




function validateOptions(filenames, options)
	% Make sure that mutually exclusive options aren't provided.
	fieldCount = 0;
	if (~isempty(options.basefile))
		fieldCount = fieldCount + 1;
	end
	if (~isempty(options.expvals))
		fieldCount = fieldCount + 1;
	end
	if (~isempty(options.relexp))
		fieldCount = fieldCount + 1;
	end
	if (fieldCount > 1)
		error('Images:makehdr:tooManyExposureParameters', ...
		      'Only one of the following parameters is allowed: ''BaseFile'', ''ExposureValues'', or ''RelativeExposure''.')
	end
	% Make sure that the correct number of exposure-related values are given.
	if (~isempty(options.expvals) && (numel(options.expvals) ~= numel(filenames)))
		error('Images:makehdr:wrongExposureValuesCount', ...
		      'The number of ''ExposureValues'' elements must match the number of files.')
	elseif (~isempty(options.relexp) && (numel(options.relexp) ~= numel(filenames)))
		error('Images:makehdr:wrongRelativeExposureCount', ...
		      'The number of ''RelativeExposure'' elements must match the number of files.')
	end




function [hdr, counts] = makeContainers(meta)
	% Create a floating point accumulator for the final HDR image and a counter
	% for the number of contributing images.

	metaWidth = length(meta(1,:,:));
	metaHeight = length(meta(:,1,:));

	hdr = zeros(metaHeight, metaWidth, 3, 'single');
	counts = zeros(metaHeight, metaWidth, 3, 'single');



