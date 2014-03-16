#ifndef VEHICLESIM_OPTIONBASE
#define VEHICLESIM_OPTIONBASE

namespace vlr
{
	template <typename T>
	class OptionBase
	{
	public:
		OptionBase(T* base)
			: _base(base)
		{

		}

		void setBase(T* base)
		{
			_base = base;
		}

	protected:
		T* _base;
	};
}

#endif /* VEHICLESIM_OPTIONBASE */
