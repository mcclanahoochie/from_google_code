--
-- Quick Export Script to Swap Red and Blue Channels in Gimp
--   by Stu Fisher http://q3f.org
--
-- modified to use ImageMagick
--   by Chris McClanahan http://mcclanahoochie.com
--   http://mcclanahoochie.com/blog/2013/01/lightroom-plugins-infrared-and-pyrdetail
--

local LrTasks = import 'LrTasks'
local LrApplication = import 'LrApplication'
local LrLogger = import 'LrLogger'

local myLogger = LrLogger('RBSwap')
myLogger:enable("print")

exportServiceProvider = {}
exportServiceProvider.canExportVideo = false

function exportServiceProvider.processRenderedPhotos(functionContext, exportContext)
   
   local exportSession = exportContext.exportSession
   local exportSettings = exportContext.propertyTable
   local nPhotos = exportSession:countRenditions()
   local progressScope = exportContext:configureProgress {
      title =
         nPhotos > 1 and LOC("$$$/RBSwap/Publish/Progress=Exporting ^1 photos to RB swap", nPhotos)
         or              LOC "$$$/RBSwap/Publish/Progress/One=Exporting one photo to RB swap",
   }
   
   for i, rendition in exportContext:renditions { stopIfCanceled = true } do		
      progressScope:setPortionComplete((i - 1) / nPhotos)
      if not rendition.wasSkipped then
         local success, pathOrMessage = rendition:waitForRender()
         progressScope:setPortionComplete((i - 0.5) / nPhotos)
         if progressScope:isCanceled() then break end
         if success then
            local filePath = assert(pathOrMessage)
            -- choose app, configure bin paths --
            --result = LrTasks.execute("/Applications/Gimp-2.8.app/Contents/Resources/bin/gimp-2.8 -i -b '(red_blue_swap \"" .. filePath .. "\")' -b '(gimp-quit 0)'")
            if WIN_ENV then
               -- windows support thanks to Frank Firsching 
               cmd = "cmd.exe /c convert.exe"
            else
               cmd = "/opt/local/bin/convert"
            end
            local cmd = cmd .. " " .. filePath .. " -set colorspace RGB -separate -swap 0,2 -combine " .. filePath
            
            result = LrTasks.execute(cmd)
            -- no need to trigger import. Done automatically by Lightroom as
            -- specified in presets/RBSwap.lrtemplate
         end
      end
   end

   progressScope:done()

end

return exportServiceProvider
