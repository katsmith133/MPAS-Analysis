# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import netCDF4
import os
from functools import partial

from ..shared.constants.constants import m3ps_to_Sv
from ..shared.plot.plotting import plot_vertical_section,\
    timeseries_analysis_plot, setup_colormap

from ..shared.io.utility import build_config_full_path, make_directories

from ..shared.io import open_mpas_dataset

from ..shared.timekeeping.utility import get_simulation_start_time, \
    days_to_datetime

from ..shared import AnalysisTask

from ..shared.time_series import MpasTimeSeriesTask
from ..shared.html import write_image_xml


class StreamfunctionMOC(AnalysisTask):  # {{{
    '''
    Computation and plotting of model meridional overturning circulation.
    Will eventually support:

        * MOC streamfunction, post-processed (currently supported)
        * MOC streamfunction, from MOC analysis member
        * MOC time series (max value at 24.5N), post-processed
        * MOC time series (max value at 24.5N), from MOC analysis member

    Attributes
    ----------

    mpasClimatologyTask : ``MpasClimatologyTask``
        The task that produced the climatology to be remapped and plotted

    Authors
    -------
    Milena Veneziani, Mark Petersen, Phillip Wolfram, Xylar Asay-Davis
    '''

    def __init__(self, config, mpasClimatologyTask):  # {{{
        '''
        Construct the analysis task.

        Parameters
        ----------
        config :  instance of MpasAnalysisConfigParser
            Contains configuration options

        mpasClimatologyTask : ``MpasClimatologyTask``
            The task that produced the climatology to be remapped and plotted

        Authors
        -------
        Xylar Asay-Davis

        '''
        # first, call the constructor from the base class (AnalysisTask)
        super(StreamfunctionMOC, self).__init__(
            config=config,
            taskName='streamfunctionMOC',
            componentName='ocean',
            tags=['streamfunction', 'moc', 'climatology', 'timeSeries'])

        self.mpasClimatologyTask = mpasClimatologyTask
        self.run_after(mpasClimatologyTask)

        self.mpasTimeSeriesTask = MpasTimeSeriesTask(
                config=config,  componentName=self.componentName,
                taskName=self.taskName, subtaskName='mpasTimeSeries')
        self.run_after(self.mpasTimeSeriesTask)
        # }}}

    def setup_and_check(self):  # {{{
        '''
        Perform steps to set up the analysis and check for errors in the setup.

        Raises
        ------
        ValueError
            if timeSeriesStatsMonthly is not enabled in the MPAS run

        Authors
        -------
        Xylar Asay-Davis
        '''

        # first, call setup_and_check from the base class (AnalysisTask),
        # which will perform some common setup, including storing:
        #     self.runDirectory , self.historyDirectory, self.plotsDirectory,
        #     self.namelist, self.runStreams, self.historyStreams,
        #     self.calendar
        super(StreamfunctionMOC, self).setup_and_check()

        self.startYearClimo = self.mpasClimatologyTask.startYear
        self.startDateClimo = self.mpasClimatologyTask.startDate
        self.endYearClimo = self.mpasClimatologyTask.endYear
        self.endDateClimo = self.mpasClimatologyTask.endDate

        config = self.config

        self.mocAnalysisMemberEnabled = self.check_analysis_enabled(
            analysisOptionName='config_am_mocstreamfunction_enable',
            raiseException=False)

        self.startDateTseries = config.get('timeSeries', 'startDate')
        self.endDateTseries = config.get('timeSeries', 'endDate')
        self.startYearTseries = config.getint('timeSeries', 'startYear')
        self.endYearTseries = config.getint('timeSeries', 'endYear')

        self.sectionName = 'streamfunctionMOC'

        variableList = ['timeMonthly_avg_normalVelocity',
                        'timeMonthly_avg_vertVelocityTop']

        self.mpasClimatologyTask.add_variables(variableList=variableList,
                                               seasons=['ANN'])
        self.mpasTimeSeriesTask.add_variables(variableList=variableList)

        self.xmlFileNames = []
        self.filePrefixes = {}

        mainRunName = config.get('runs', 'mainRunName')

        regions = ['Global'] + config.getExpression(self.sectionName,
                                                    'regionNames')

        for region in regions:
            filePrefix = 'moc{}_{}_years{:04d}-{:04d}'.format(
                    region, mainRunName,
                    self.startYearClimo, self.endYearClimo)

            self.xmlFileNames.append('{}/{}.xml'.format(self.plotsDirectory,
                                                        filePrefix))
            self.filePrefixes[region] = filePrefix

        filePrefix = 'mocTimeseries_{}'.format(mainRunName)
        self.xmlFileNames.append('{}/{}.xml'.format(self.plotsDirectory,
                                                    filePrefix))
        self.filePrefixes['timeSeries'] = filePrefix

        # }}}

    def run_task(self):  # {{{
        '''
        Process MOC analysis member data if available, or compute MOC at
        post-processing if not. Plots streamfunction climatolgoical sections
        as well as time series of max Atlantic MOC at 26.5N (latitude of
        RAPID MOC Array).

        Authors
        -------
        Milena Veneziani, Mark Petersen, Phillip J. Wolfram, Xylar Asay-Davis
        '''

        self.logger.info("\nPlotting streamfunction of Meridional Overturning "
                         "Circulation (MOC)...")

        config = self.config

        # **** Compute MOC ****
        # Check whether MOC Analysis Member is enabled
        if self.mocAnalysisMemberEnabled:
            # Add a moc_analisysMember_processing
            self.logger.info('*** MOC Analysis Member is on ***')
            # (mocDictClimo, mocDictTseries) = \
            #     self._compute_moc_analysismember(config, streams, calendar,
            #                                      sectionName, dictClimo,
            #                                      dictTseries)

            # delete the following 3 lines after analysis of the MOC AM is
            # supported
            self.logger.info('...but not yet supported. Using offline MOC')
            self._compute_moc_climo_postprocess()
            dsMOCTimeSeries = self._compute_moc_time_series_postprocess()
        else:
            self._compute_moc_climo_postprocess()
            dsMOCTimeSeries = self._compute_moc_time_series_postprocess()

        # **** Plot MOC ****
        # Define plotting variables
        mainRunName = config.get('runs', 'mainRunName')
        movingAveragePoints = config.getint(self.sectionName,
                                            'movingAveragePoints')
        movingAveragePointsClimatological = config.getint(
                self.sectionName, 'movingAveragePointsClimatological')
        colorbarLabel = '[Sv]'
        xLabel = 'latitude [deg]'
        yLabel = 'depth [m]'

        for region in self.regionNames:
            self.logger.info('   Plot climatological {} MOC...'.format(region))
            title = '{} MOC (ANN, years {:04d}-{:04d})\n {}'.format(
                     region, self.startYearClimo,
                     self.endYearClimo,
                     mainRunName)
            filePrefix = self.filePrefixes[region]
            figureName = '{}/{}.png'.format(self.plotsDirectory, filePrefix)
            contourLevels = \
                config.getExpression(self.sectionName,
                                     'contourLevels{}'.format(region),
                                     usenumpyfunc=True)
            (colormapName, colorbarLevels) = setup_colormap(config,
                                                            self.sectionName,
                                                            suffix=region)

            x = self.lat[region]
            y = self.depth
            z = self.moc[region]
            plot_vertical_section(config, x, y, z, colormapName,
                                  colorbarLevels, contourLevels, colorbarLabel,
                                  title, xLabel, yLabel, figureName,
                                  N=movingAveragePointsClimatological)

            caption = '{} Meridional Overturning Streamfunction'.format(region)
            write_image_xml(
                config=config,
                filePrefix=filePrefix,
                componentName='Ocean',
                componentSubdirectory='ocean',
                galleryGroup='Meridional Overturning Streamfunction',
                groupLink='moc',
                thumbnailDescription=region,
                imageDescription=caption,
                imageCaption=caption)  # }}}

        # Plot time series
        self.logger.info('   Plot time series of max Atlantic MOC at 26.5N...')
        xLabel = 'Time [years]'
        yLabel = '[Sv]'
        title = 'Max Atlantic MOC at $26.5^\circ$N\n {}'.format(mainRunName)
        filePrefix = self.filePrefixes['timeSeries']

        figureName = '{}/{}.png'.format(self.plotsDirectory, filePrefix)

        timeseries_analysis_plot(config, [dsMOCTimeSeries.mocAtlantic26],
                                 movingAveragePoints, title,
                                 xLabel, yLabel, figureName,
                                 lineStyles=['k-'], lineWidths=[2],
                                 legendText=[None], calendar=self.calendar)

        caption = u'Time Series of maximum Meridional Overturning ' \
                  u'Circulation at 26.5°N'
        write_image_xml(
            config=config,
            filePrefix=filePrefix,
            componentName='Ocean',
            componentSubdirectory='ocean',
            galleryGroup='Meridional Overturning Streamfunction',
            groupLink='moc',
            thumbnailDescription='Time Series',
            imageDescription=caption,
            imageCaption=caption)  # }}}

        # }}}

    def _load_mesh(self):  # {{{
        # Load mesh related variables
        try:
            restartFile = self.runStreams.readpath('restart')[0]
        except ValueError:
            raise IOError('No MPAS-O restart file found: need at least one '
                          'restart file for MOC calculation')
        ncFile = netCDF4.Dataset(restartFile, mode='r')
        dvEdge = ncFile.variables['dvEdge'][:]
        areaCell = ncFile.variables['areaCell'][:]
        refBottomDepth = ncFile.variables['refBottomDepth'][:]
        latCell = np.rad2deg(ncFile.variables['latCell'][:])
        ncFile.close()
        nVertLevels = len(refBottomDepth)
        refTopDepth = np.zeros(nVertLevels+1)
        refTopDepth[1:nVertLevels+1] = refBottomDepth[0:nVertLevels]
        refLayerThickness = np.zeros(nVertLevels)
        refLayerThickness[0] = refBottomDepth[0]
        refLayerThickness[1:nVertLevels] = \
            (refBottomDepth[1:nVertLevels] -
             refBottomDepth[0:nVertLevels-1])

        return dvEdge, areaCell, refBottomDepth, latCell, nVertLevels, \
            refTopDepth, refLayerThickness
        # }}}

    def _compute_moc_climo_postprocess(self):  # {{{

        '''compute mean MOC streamfunction as a post-process'''

        config = self.config

        dvEdge, areaCell, refBottomDepth, latCell, nVertLevels, \
            refTopDepth, refLayerThickness = self._load_mesh()

        self.regionNames = config.getExpression(self.sectionName,
                                                'regionNames')

        # Load basin region related variables and save them to dictionary
        # NB: The following will need to change with new regional mapping files
        regionMaskFiles = config.get(self.sectionName, 'regionMaskFiles')
        if not os.path.exists(regionMaskFiles):
            raise IOError('Regional masking file for MOC calculation '
                          'does not exist')
        iRegion = 0
        self.dictRegion = {}
        for region in self.regionNames:
            self.logger.info('\n  Reading region and transect mask for '
                             '{}...'.format(region))
            ncFileRegional = netCDF4.Dataset(regionMaskFiles, mode='r')
            maxEdgesInTransect = \
                ncFileRegional.dimensions['maxEdgesInTransect'].size
            transectEdgeMaskSigns = \
                ncFileRegional.variables['transectEdgeMaskSigns'][:, iRegion]
            transectEdgeGlobalIDs = \
                ncFileRegional.variables['transectEdgeGlobalIDs'][iRegion, :]
            regionCellMask = \
                ncFileRegional.variables['regionCellMasks'][:, iRegion]
            ncFileRegional.close()
            iRegion += 1

            indRegion = np.where(regionCellMask == 1)
            self.dictRegion[region] = {
                'indices': indRegion,
                'cellMask': regionCellMask,
                'maxEdgesInTransect': maxEdgesInTransect,
                'transectEdgeMaskSigns': transectEdgeMaskSigns,
                'transectEdgeGlobalIDs': transectEdgeGlobalIDs}
        # Add Global regionCellMask=1 everywhere to make the algorithm
        # for the global moc similar to that of the regional moc

        self.dictRegion['Global'] = {
                'cellMask': np.ones(np.size(latCell))}
        self.regionNames.append('Global')

        # Compute and plot annual climatology of MOC streamfunction
        self.logger.info('\n  Compute and/or plot post-processed MOC '
                         'climatological streamfunction...')
        outputDirectory = build_config_full_path(config, 'output',
                                                 'mpasClimatologySubdirectory')

        make_directories(outputDirectory)

        outputFileClimo = '{}/mocStreamfunction_years{:04d}-{:04d}.nc'.format(
                           outputDirectory, self.startYearClimo,
                           self.endYearClimo)
        if not os.path.exists(outputFileClimo):
            self.logger.info('   Load data...')

            climatologyFileName = self.mpasClimatologyTask.get_file_name(
                season='ANN')
            annualClimatology = xr.open_dataset(climatologyFileName)
            # rename some variables for convenience
            annualClimatology = annualClimatology.rename(
                    {'timeMonthly_avg_normalVelocity': 'avgNormalVelocity',
                     'timeMonthly_avg_vertVelocityTop': 'avgVertVelocityTop'})
            annualClimatology = annualClimatology.isel(Time=0)

            # Convert to numpy arrays
            # (can result in a memory error for large array size)
            horizontalVel = annualClimatology.avgNormalVelocity.values
            verticalVel = annualClimatology.avgVertVelocityTop.values
            velArea = verticalVel * areaCell[:, np.newaxis]

            # Create dictionary for MOC climatology (NB: need this form
            # in order to convert it to xarray dataset later in the script)
            self.depth = refTopDepth
            self.lat = {}
            self.moc = {}
            for region in self.regionNames:
                self.logger.info('   Compute {} MOC...'.format(region))
                self.logger.info('    Compute transport through region '
                                 'southern transect...')
                if region == 'Global':
                    transportZ = np.zeros(nVertLevels)
                else:
                    maxEdgesInTransect = \
                        self.dictRegion[region]['maxEdgesInTransect']
                    transectEdgeGlobalIDs = \
                        self.dictRegion[region]['transectEdgeGlobalIDs']
                    transectEdgeMaskSigns = \
                        self.dictRegion[region]['transectEdgeMaskSigns']
                    transportZ = self._compute_transport(maxEdgesInTransect,
                                                         transectEdgeGlobalIDs,
                                                         transectEdgeMaskSigns,
                                                         nVertLevels, dvEdge,
                                                         refLayerThickness,
                                                         horizontalVel)

                regionCellMask = self.dictRegion[region]['cellMask']
                latBinSize = \
                    config.getExpression(self.sectionName,
                                         'latBinSize{}'.format(region))
                if region == 'Global':
                    latBins = np.arange(-90.0, 90.1, latBinSize)
                else:
                    indRegion = self.dictRegion[region]['indices']
                    latBins = latCell[indRegion]
                    latBins = np.arange(np.amin(latBins),
                                        np.amax(latBins)+latBinSize,
                                        latBinSize)
                mocTop = self._compute_moc(latBins, nVertLevels, latCell,
                                           regionCellMask, transportZ, velArea)

                # Store computed MOC to dictionary
                self.lat[region] = latBins
                self.moc[region] = mocTop

            # Save to file
            self.logger.info('   Save global and regional MOC to file...')
            ncFile = netCDF4.Dataset(outputFileClimo, mode='w')
            # create dimensions
            ncFile.createDimension('nz', len(refTopDepth))
            for region in self.regionNames:
                latBins = self.lat[region]
                mocTop = self.moc[region]
                ncFile.createDimension('nx{}'.format(region), len(latBins))
                # create variables
                x = ncFile.createVariable('lat{}'.format(region), 'f4',
                                          ('nx{}'.format(region),))
                x.description = 'latitude bins for MOC {}'\
                                ' streamfunction'.format(region)
                x.units = 'degrees (-90 to 90)'
                y = ncFile.createVariable('moc{}'.format(region), 'f4',
                                          ('nz', 'nx{}'.format(region)))
                y.description = 'MOC {} streamfunction, annual'\
                                ' climatology'.format(region)
                y.units = 'Sv (10^6 m^3/s)'
                # save variables
                x[:] = latBins
                y[:, :] = mocTop
            depth = ncFile.createVariable('depth', 'f4', ('nz',))
            depth.description = 'depth'
            depth.units = 'meters'
            depth[:] = refTopDepth
            ncFile.close()
        else:
            # Read from file
            self.logger.info('   Read previously computed MOC streamfunction '
                             'from file...')
            ncFile = netCDF4.Dataset(outputFileClimo, mode='r')
            self.depth = ncFile.variables['depth'][:]
            self.lat = {}
            self.moc = {}
            for region in self.regionNames:
                self.lat[region] = ncFile.variables['lat{}'.format(region)][:]
                self.moc[region] = \
                    ncFile.variables['moc{}'.format(region)][:, :]
            ncFile.close()
        # }}}

    def _compute_moc_time_series_postprocess(self):  # {{{
        '''compute MOC time series as a post-process'''

        # Compute and plot time series of Atlantic MOC at 26.5N (RAPID array)
        self.logger.info('\n  Compute and/or plot post-processed Atlantic MOC '
                         'time series...')
        self.logger.info('   Load data...')

        config = self.config

        variableList = ['timeMonthly_avg_normalVelocity',
                        'timeMonthly_avg_vertVelocityTop']

        dvEdge, areaCell, refBottomDepth, latCell, nVertLevels, \
            refTopDepth, refLayerThickness = self._load_mesh()

        fileName = self.mpasTimeSeriesTask.outputFile
        ds = open_mpas_dataset(
            fileName=fileName,
            calendar=self.calendar,
            variableList=variableList,
            startDate=self.startDateTseries,
            endDate=self.endDateTseries)
        latAtlantic = self.lat['Atlantic']
        dLat = latAtlantic - 26.5
        indlat26 = np.where(dLat == np.amin(np.abs(dLat)))

        dictRegion = self.dictRegion['Atlantic']
        maxEdgesInTransect = dictRegion['maxEdgesInTransect']
        transectEdgeGlobalIDs = dictRegion['transectEdgeGlobalIDs']
        transectEdgeMaskSigns = dictRegion['transectEdgeMaskSigns']
        regionCellMask = dictRegion['cellMask']

        outputDirectory = build_config_full_path(config, 'output',
                                                 'timeseriesSubdirectory')
        try:
            os.makedirs(outputDirectory)
        except OSError:
            pass

        outputFileTseries = '{}/mocTimeSeries.nc'.format(outputDirectory)

        continueOutput = os.path.exists(outputFileTseries)
        if continueOutput:
            self.logger.info('   Read in previously computed MOC time series')

        dsMOCTimeSeries = self._compute_moc_time_series(
                ds, areaCell, latCell, indlat26, maxEdgesInTransect,
                transectEdgeGlobalIDs, transectEdgeMaskSigns,  nVertLevels,
                dvEdge, refLayerThickness, latAtlantic, regionCellMask)

        return dsMOCTimeSeries  # }}}

    def _compute_moc_time_series(self, ds, areaCell, latCell, indlat26,
                                 maxEdgesInTransect,
                                 transectEdgeGlobalIDs,
                                 transectEdgeMaskSigns, nVertLevels,
                                 dvEdge, refLayerThickness, latAtlantic,
                                 regionCellMask):
        # computes a subset of the MOC time series

        times = ds.Time.values
        mocRegion = np.zeros(times.shape)

        for timeIndex, time in enumerate(times):
            dsLocal = ds.isel(Time=timeIndex)
            date = days_to_datetime(time, calendar=self.calendar)

            self.logger.info('     date: {:04d}-{:02d}'.format(date.year,
                                                               date.month))

            horizontalVel = dsLocal.timeMonthly_avg_normalVelocity.values
            verticalVel = dsLocal.timeMonthly_avg_vertVelocityTop.values
            velArea = verticalVel * areaCell[:, np.newaxis]
            transportZ = self._compute_transport(maxEdgesInTransect,
                                                 transectEdgeGlobalIDs,
                                                 transectEdgeMaskSigns,
                                                 nVertLevels, dvEdge,
                                                 refLayerThickness,
                                                 horizontalVel)
            mocTop = self._compute_moc(latAtlantic, nVertLevels, latCell,
                                       regionCellMask, transportZ, velArea)
            mocRegion[timeIndex] = np.amax(mocTop[:, indlat26])

        description = 'Max MOC Atlantic streamfunction nearest to RAPID ' \
            'Array latitude (26.5N)'

        dictonary = {'dims': ['Time'],
                     'coords': {'Time':
                                {'dims': ('Time'),
                                 'data': times,
                                 'attrs': {'units': 'days since 0001-01-01'}}},
                     'data_vars': {'mocAtlantic26':
                                   {'dims': ('Time'),
                                    'data': mocRegion,
                                    'attrs': {'units': 'Sv (10^6 m^3/s)',
                                              'description': description}}}}
        dsMOC = xr.Dataset.from_dict(dictonary)
        return dsMOC

    # def _compute_moc_analysismember(self):
    #
    #     return

    def _compute_transport(self, maxEdgesInTransect, transectEdgeGlobalIDs,
                           transectEdgeMaskSigns, nz, dvEdge,
                           refLayerThickness, horizontalVel):  # {{{

        '''compute mass transport across southern transect of ocean basin'''

        transportZEdge = np.zeros([nz, maxEdgesInTransect])
        for i in range(maxEdgesInTransect):
            if transectEdgeGlobalIDs[i] == 0:
                break
            # subtract 1 because of python 0-indexing
            iEdge = transectEdgeGlobalIDs[i] - 1
            transportZEdge[:, i] = horizontalVel[iEdge, :] * \
                transectEdgeMaskSigns[iEdge, np.newaxis] * \
                dvEdge[iEdge, np.newaxis] * \
                refLayerThickness[np.newaxis, :]
        transportZ = transportZEdge.sum(axis=1)
        return transportZ  # }}}

    def _compute_moc(self, latBins, nz, latCell, regionCellMask, transportZ,
                     velArea):  # {{{

        '''compute meridionally integrated MOC streamfunction'''

        mocTop = np.zeros([np.size(latBins), nz+1])
        mocTop[0, range(1, nz+1)] = transportZ.cumsum()
        for iLat in range(1, np.size(latBins)):
            indlat = np.logical_and(np.logical_and(
                         regionCellMask == 1, latCell >= latBins[iLat-1]),
                         latCell < latBins[iLat])
            mocTop[iLat, :] = mocTop[iLat-1, :] + \
                velArea[indlat, :].sum(axis=0)
        # convert m^3/s to Sverdrup
        mocTop = mocTop * m3ps_to_Sv
        mocTop = mocTop.T
        return mocTop  # }}}

# }}}

# vim: foldmethod=marker ai ts=4 sts=4 et sw=4 ft=python
