--
-- Quick Export Script to Swap Red and Blue Channels in Gimp
--   by Stu Fisher http://q3f.org
--
-- modified to use ImageMagick
--   by Chris McClanahan http://mcclanahoochie.com
--

return {
	LrSdkVersion = 4.0,
	LrSdkMinimumVersion = 3.0,
	LrToolkitIdentifier = 'com.adobe.lightroom.export.rbswap',
	LrPluginName = LOC "$$$/YAPB/PluginName=Red Blue Swap",
	LrExportServiceProvider = {
		title = LOC "$$$/YAPB/YAPB-title=RBSwap",
		file = 'RBSwapExportServiceProvider.lua',
                builtInPresetsDir = 'presets',
	},
	VERSION = {
           major=1, minor=1, revision=0, build=2,
        },
}
